"""
AllToAllV algorithm templates.

Each template defines a fundamentally different approach to implementing AllToAllV.
The search optimizes parameters *within* each template, then compares across templates.

Templates:
  permute_ring     - Sequential collective_permute with optimized distance ordering
  allgather_slice  - AllGather full buffer + local slice per rank
  hierarchical     - Intra-device (free) then inter-device permute
  pairwise         - Paired bidirectional exchanges (N/2 pairs per round)
  hybrid_ag_perm   - AllGather for nearby ranks, permute for distant ranks
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TemplateConfig:
    """Configuration for an algorithm template instance."""
    template: str
    params: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.template


# ============================================================
# Template: permute_ring
# ============================================================
# The current approach. Sequential collective_permute steps.
# Search space: ordering of distances [1..world-1]

def permute_ring_default_params(world=32):
    return {"schedule": list(range(1, world))}


def permute_ring_search_space(world=32):
    """Describe the search space for GA/SA."""
    return {
        "type": "permutation",
        "elements": list(range(1, world)),
        "param_key": "schedule",
    }


# ============================================================
# Template: allgather_slice
# ============================================================
# AllGather entire send buffer across all ranks, then local slice.
# No schedule to optimize — fixed algorithm. But we can vary:
#   - chunk_factor: split the allgather into N chunks for pipelining
#   - gather_dim: dimension along which to gather

def allgather_slice_default_params(world=32):
    return {"chunk_factor": 1}  # 1 = no chunking


def allgather_slice_search_space(world=32):
    return {
        "type": "integer_choice",
        "choices": [1, 2, 4, 8],
        "param_key": "chunk_factor",
    }


# ============================================================
# Template: hierarchical
# ============================================================
# Two-level approach exploiting the 2-core-per-device structure:
#   Level 1: Intra-device exchange (core 2i <-> 2i+1) — essentially free
#   Level 2: Inter-device permute with optimized schedule on 16 devices
# This reduces the permute steps from 31 to 15 (inter-device only).

def hierarchical_default_params(world=32, num_devices=16):
    return {
        "inter_schedule": list(range(1, num_devices)),  # 15 device-level distances
    }


def hierarchical_search_space(world=32, num_devices=16):
    return {
        "type": "permutation",
        "elements": list(range(1, num_devices)),
        "param_key": "inter_schedule",
    }


# ============================================================
# Template: pairwise
# ============================================================
# Each round: pair all N ranks into N/2 pairs, each pair exchanges data.
# All pairs in a round run concurrently.
# For N=32, need 31 rounds. Each round is defined by a perfect matching.
# Search space: ordering of the 31 matchings.
#
# We generate matchings using the "round-robin tournament" algorithm:
# Fix rank 0, rotate ranks 1..N-1. At round k, pair rank 0 with rank k,
# and pair the remaining symmetrically.

def _generate_matchings(world=32):
    """Generate all N-1 perfect matchings using round-robin tournament."""
    matchings = []
    ranks = list(range(world))
    for round_idx in range(world - 1):
        pairs = []
        # Standard round-robin: fix position 0, rotate the rest
        rotated = [ranks[0]] + [ranks[1 + (round_idx + i) % (world - 1)] for i in range(world - 1)]
        for i in range(world // 2):
            pairs.append((rotated[i], rotated[world - 1 - i]))
        matchings.append(pairs)
    return matchings


def pairwise_default_params(world=32):
    matchings = _generate_matchings(world)
    return {
        "round_order": list(range(len(matchings))),
        "_matchings": matchings,
    }


def pairwise_search_space(world=32):
    return {
        "type": "permutation",
        "elements": list(range(world - 1)),
        "param_key": "round_order",
    }


# ============================================================
# Template: hybrid_ag_perm
# ============================================================
# Use AllGather for ranks within hop_threshold hops,
# use collective_permute for ranks beyond hop_threshold.
# Parameters:
#   - hop_threshold: 1 or 2 (which ranks use allgather vs permute)
#   - permute_schedule: ordering for the permute distances

def hybrid_default_params(topology, world=32):
    near_distances = []
    far_distances = []
    for d in range(1, world):
        avg_hops = sum(topology.rank_hops(r, (r + d) % world) for r in range(world)) / world
        if avg_hops <= 1.0:
            near_distances.append(d)
        else:
            far_distances.append(d)
    return {
        "hop_threshold": 1.0,
        "near_distances": near_distances,
        "far_distances": far_distances,
        "permute_schedule": far_distances.copy(),
    }


def hybrid_search_space(topology, world=32):
    params = hybrid_default_params(topology, world)
    return {
        "type": "permutation",
        "elements": params["far_distances"],
        "param_key": "permute_schedule",
    }


# ============================================================
# Template: multinode_hierarchical
# ============================================================
# Three-level approach for multi-node clusters:
#   Level 1: Intra-device exchange (free, shared HBM)
#   Level 2: Intra-node inter-device permute (NeuronLink, fast)
#   Level 3: Inter-node exchange (EFA, slow)

def multinode_hierarchical_default_params(world=64, num_devices=32,
                                          num_nodes=2, **kwargs):
    devices_per_node = num_devices // max(num_nodes, 1)
    return {
        "intra_node_schedule": list(range(1, devices_per_node)),
        "inter_node_schedule": list(range(1, num_nodes)),
    }


def multinode_hierarchical_search_space(world=64, num_devices=32,
                                        num_nodes=2, **kwargs):
    devices_per_node = num_devices // max(num_nodes, 1)
    return {
        "type": "nested_permutation",
        "intra": {
            "type": "permutation",
            "elements": list(range(1, devices_per_node)),
            "param_key": "intra_node_schedule",
        },
        "inter": {
            "type": "permutation",
            "elements": list(range(1, num_nodes)),
            "param_key": "inter_node_schedule",
        },
    }


# ============================================================
# Template: node_allgather
# ============================================================
# Two-level AllGather for multi-node clusters:
#   Phase 1: AllGather within each node (32 ranks, NeuronLink ring)
#   Phase 2: AllGather across nodes (EFA ring)
# Amplification: 32x (intra-node) + Nx (inter-node)

def node_allgather_default_params(world=64, **kwargs):
    return {}


def node_allgather_search_space(world=64, **kwargs):
    return {"type": "none"}


# ============================================================
# Registry
# ============================================================

TEMPLATES = {
    "permute_ring": {
        "description": "Sequential collective_permute with optimized distance ordering",
        "default_params": permute_ring_default_params,
        "search_space": permute_ring_search_space,
    },
    "allgather_slice": {
        "description": "AllGather full buffer + local slice (low-latency for small messages)",
        "default_params": allgather_slice_default_params,
        "search_space": allgather_slice_search_space,
    },
    "hierarchical": {
        "description": "2-level: intra-device (free) then inter-device permute (15 steps vs 31)",
        "default_params": hierarchical_default_params,
        "search_space": hierarchical_search_space,
    },
    "pairwise": {
        "description": "Paired bidirectional exchanges (N/2 concurrent pairs per round)",
        "default_params": pairwise_default_params,
        "search_space": pairwise_search_space,
    },
    "hybrid_ag_perm": {
        "description": "AllGather for nearby ranks, permute for distant ranks",
        "default_params": None,  # needs topology
        "search_space": None,    # needs topology
    },
    "fused_alltoall": {
        "description": "Single xm.all_to_all collective (1 dispatch, pack/unpack with max_chunk padding)",
        "default_params": lambda world=32, **kw: {},
        "search_space": lambda world=32, **kw: {"type": "none"},
    },
    "allgather_reduce_scatter": {
        "description": "all_gather(counts) + reduce_scatter(data): 2 dispatches, minimal network traffic",
        "default_params": lambda world=32, **kw: {},
        "search_space": lambda world=32, **kw: {"type": "none"},
    },
"multinode_hierarchical": {
        "description": "3-level: intra-device (free) -> intra-node torus (NeuronLink) -> inter-node (EFA)",
        "default_params": multinode_hierarchical_default_params,
        "search_space": multinode_hierarchical_search_space,
    },
    "node_allgather": {
        "description": "2-level AllGather: intra-node gather (NeuronLink) + inter-node gather (EFA)",
        "default_params": node_allgather_default_params,
        "search_space": node_allgather_search_space,
    },
}


def list_templates():
    for name, info in TEMPLATES.items():
        print(f"  {name:20s} - {info['description']}")
