"""
Trainium trn1.32xlarge topology model.

Physical topology from neuron-ls:
- 16 NeuronDevices, each with 2 NeuronCores (32 cores total)
- Devices connected in a 2D 4x4 torus via NeuronLink
- Each device has 4 bidirectional links (~192 GB/s per link)
- Intra-device core-to-core communication is essentially free (shared HBM)

We model at the NeuronCore level (32 ranks) since torchrun uses --nproc_per_node=32.
"""

import math
from collections import defaultdict


# Actual adjacency from neuron-ls on trn1.32xlarge
DEVICE_ADJACENCY = {
    0:  [12, 3, 4, 1],
    1:  [13, 0, 5, 2],
    2:  [14, 1, 6, 3],
    3:  [15, 2, 7, 0],
    4:  [0, 7, 8, 5],
    5:  [1, 4, 9, 6],
    6:  [2, 5, 10, 7],
    7:  [3, 6, 11, 4],
    8:  [4, 11, 12, 9],
    9:  [5, 8, 13, 10],
    10: [6, 9, 14, 11],
    11: [7, 10, 15, 8],
    12: [8, 15, 0, 13],
    13: [9, 12, 1, 14],
    14: [10, 13, 2, 15],
    15: [11, 14, 3, 12],
}

NUM_DEVICES = 16
CORES_PER_DEVICE = 2
NUM_CORES = NUM_DEVICES * CORES_PER_DEVICE  # 32


class Link:
    """Bidirectional NeuronLink between two devices."""

    def __init__(self, bandwidth_GBps, latency_us):
        self.bandwidth = bandwidth_GBps * 1e9  # bytes/sec
        self.latency = latency_us * 1e-6       # seconds
        self.next_free_fwd = 0.0
        self.next_free_bwd = 0.0

    def reset(self):
        self.next_free_fwd = 0.0
        self.next_free_bwd = 0.0

    def transmit(self, src_dev, dst_dev, bytes_, current_time, link_key):
        """Transmit bytes from src to dst. Direction determined by link_key ordering."""
        tx_time = bytes_ / self.bandwidth
        is_forward = (src_dev == link_key[0])
        if is_forward:
            start = max(current_time, self.next_free_fwd)
            finish = start + self.latency + tx_time
            self.next_free_fwd = finish
        else:
            start = max(current_time, self.next_free_bwd)
            finish = start + self.latency + tx_time
            self.next_free_bwd = finish
        return finish


class TrainiumTopology:
    """
    Models a NeuronLink topology (e.g. trn1.32xlarge 4x4 torus).

    Topology parameters (num_devices, cores_per_device, device_adjacency) default
    to the trn1.32xlarge 4x4 torus but can be overridden for other configurations.
    The profiler agent (Phase 0) discovers these at runtime.

    Parameters:
        link_bandwidth_GBps: per-link bandwidth in GB/s (default ~192 GB/s per NeuronLink)
        link_latency_us: per-hop latency in microseconds
        num_devices: number of NeuronDevices (default: 16 for trn1.32xlarge)
        cores_per_device: NeuronCores per device (default: 2 for trn1, 4 for trn2)
        device_adjacency: dict mapping device_id -> list of neighbor device_ids
    """

    def __init__(self, link_bandwidth_GBps=192.0, link_latency_us=0.5,
                 num_devices=None, cores_per_device=None, device_adjacency=None):
        self.num_devices = num_devices if num_devices is not None else NUM_DEVICES
        self.cores_per_device = cores_per_device if cores_per_device is not None else CORES_PER_DEVICE
        self.num_cores = self.num_devices * self.cores_per_device
        self.adjacency = device_adjacency if device_adjacency is not None else DEVICE_ADJACENCY
        self.link_bw = link_bandwidth_GBps
        self.link_lat = link_latency_us

        # Build links (keyed by sorted device pair)
        self.links = {}
        for dev, neighbors in self.adjacency.items():
            for nbr in neighbors:
                key = (min(dev, nbr), max(dev, nbr))
                if key not in self.links:
                    self.links[key] = Link(link_bandwidth_GBps, link_latency_us)

        # Precompute shortest paths (BFS) between all device pairs
        self._shortest_paths = {}
        for src in range(self.num_devices):
            self._shortest_paths[src] = self._bfs(src)

    def reset(self):
        for link in self.links.values():
            link.reset()

    def rank_to_device(self, rank):
        return rank // self.cores_per_device

    def device_to_ranks(self, device):
        base = device * self.cores_per_device
        return list(range(base, base + self.cores_per_device))

    def is_same_device(self, rank_a, rank_b):
        return self.rank_to_device(rank_a) == self.rank_to_device(rank_b)

    def _bfs(self, src_dev):
        """BFS to find shortest path from src_dev to all other devices."""
        visited = {src_dev: [src_dev]}
        queue = [src_dev]
        while queue:
            next_queue = []
            for node in queue:
                for nbr in self.adjacency[node]:
                    if nbr not in visited:
                        visited[nbr] = visited[node] + [nbr]
                        next_queue.append(nbr)
            queue = next_queue
        return visited

    def device_path(self, src_dev, dst_dev):
        """Return list of devices on shortest path from src to dst (inclusive)."""
        return self._shortest_paths[src_dev][dst_dev]

    def device_hops(self, src_dev, dst_dev):
        """Number of hops between two devices."""
        return len(self._shortest_paths[src_dev][dst_dev]) - 1

    def rank_hops(self, src_rank, dst_rank):
        """Number of hops between two ranks (0 if same device)."""
        return self.device_hops(
            self.rank_to_device(src_rank),
            self.rank_to_device(dst_rank)
        )

    def send(self, src_rank, dst_rank, bytes_, start_time=0.0):
        """
        Simulate sending bytes from src_rank to dst_rank.
        Returns finish time. Accounts for multi-hop routing and link contention.
        """
        src_dev = self.rank_to_device(src_rank)
        dst_dev = self.rank_to_device(dst_rank)

        if src_dev == dst_dev:
            return start_time  # intra-device, negligible

        path = self.device_path(src_dev, dst_dev)
        t = start_time
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = (min(a, b), max(a, b))
            link = self.links[key]
            t = link.transmit(a, b, bytes_, t, key)
        return t

    def hop_matrix(self):
        """Return NxN matrix of hop counts between all ranks."""
        matrix = []
        for i in range(self.num_cores):
            row = []
            for j in range(self.num_cores):
                row.append(self.rank_hops(i, j))
            matrix.append(row)
        return matrix

    def neighbor_ranks(self, rank):
        """Return ranks that are 1 hop away (including same-device peers)."""
        dev = self.rank_to_device(rank)
        result = []
        # Same-device peers (all cores on same device except self)
        for r in self.device_to_ranks(dev):
            if r != rank:
                result.append(r)
        # Neighbors via NeuronLink
        for nbr_dev in self.adjacency[dev]:
            result.extend(self.device_to_ranks(nbr_dev))
        return result

    def summary(self):
        """Print topology summary."""
        hops = self.hop_matrix()
        max_hop = max(max(row) for row in hops)
        avg_hop = sum(sum(row) for row in hops) / (self.num_cores * self.num_cores)
        print(f"Trainium trn1.32xlarge topology:")
        print(f"  Devices: {self.num_devices}, Cores: {self.num_cores}")
        print(f"  Link BW: {self.link_bw} GB/s, Latency: {self.link_lat} us")
        print(f"  Max hops: {max_hop}, Avg hops: {avg_hop:.2f}")
        print(f"  Links: {len(self.links)}")


# ======================================================================
# Multi-node cluster topology
# ======================================================================

class EFALink:
    """Models a single EFA adapter (unidirectional)."""

    def __init__(self, bandwidth_GBps, latency_us):
        self.bandwidth = bandwidth_GBps * 1e9  # bytes/sec
        self.latency = latency_us * 1e-6       # seconds
        self.next_free = 0.0

    def reset(self):
        self.next_free = 0.0

    def transmit(self, bytes_, current_time):
        tx_time = bytes_ / self.bandwidth
        start = max(current_time, self.next_free)
        finish = start + self.latency + tx_time
        self.next_free = finish
        return finish


class MultiNodeTopology:
    """
    Models a cluster of N trn1.32xlarge nodes connected via EFA.

    Intra-node: 4x4 torus of 16 NeuronDevices via NeuronLink (~192 GB/s).
    Inter-node: EFA adapters (~12.5 GB/s each, 8 per node, ~5 us latency).

    Rank mapping: global_rank = node_id * 32 + local_rank
                  local_rank  = device * 2 + core

    When num_nodes=1, behavior is identical to TrainiumTopology.
    """

    def __init__(self, num_nodes=1,
                 neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                 efa_bandwidth_GBps=12.5, efa_adapters_per_node=8,
                 efa_latency_us=5.0,
                 num_devices_per_node=None, cores_per_device=None,
                 device_adjacency=None):
        self.num_nodes = num_nodes
        _cpd = cores_per_device if cores_per_device is not None else CORES_PER_DEVICE
        _dpn = num_devices_per_node if num_devices_per_node is not None else NUM_DEVICES
        self.devices_per_node = _dpn
        self.cores_per_device = _cpd
        self.ranks_per_node = _dpn * _cpd
        self.num_devices = num_nodes * _dpn
        self.num_cores = num_nodes * _dpn * _cpd
        self.efa_bw = efa_bandwidth_GBps
        self.efa_lat = efa_latency_us
        self.efa_adapters = efa_adapters_per_node
        self.link_bw = neuronlink_bandwidth_GBps
        self.link_lat = neuronlink_latency_us

        _adj = device_adjacency if device_adjacency is not None else DEVICE_ADJACENCY

        # Per-node NeuronLink torus topologies
        self.nodes = [
            TrainiumTopology(neuronlink_bandwidth_GBps, neuronlink_latency_us,
                             num_devices=_dpn, cores_per_device=_cpd,
                             device_adjacency=_adj)
            for _ in range(num_nodes)
        ]

        # Expose single-node adjacency for prompt generation
        self.adjacency = _adj

        # EFA links: keyed by (src_node, dst_node, adapter_idx)
        # Each direction is independent (full-duplex).
        self.efa_links = {}
        for a in range(num_nodes):
            for b in range(num_nodes):
                if a == b:
                    continue
                self.efa_links[(a, b)] = [
                    EFALink(efa_bandwidth_GBps, efa_latency_us)
                    for _ in range(efa_adapters_per_node)
                ]

        # Precompute shortest paths between all device pairs
        self._shortest_paths = {}
        for src in range(self.num_devices):
            self._shortest_paths[src] = {}
            src_node = src // self.devices_per_node
            src_local = src % self.devices_per_node
            for dst in range(self.num_devices):
                dst_node = dst // self.devices_per_node
                dst_local = dst % self.devices_per_node
                if src_node == dst_node:
                    # Intra-node path: delegate to per-node torus BFS
                    local_path = self.nodes[src_node].device_path(
                        src_local, dst_local)
                    self._shortest_paths[src][dst] = [
                        src_node * self.devices_per_node + d
                        for d in local_path
                    ]
                else:
                    # Inter-node: src_device -> EFA -> dst_device
                    # Represented as [src, dst] (1 "hop" through EFA)
                    self._shortest_paths[src][dst] = [src, dst]

    def reset(self):
        for node in self.nodes:
            node.reset()
        for link_list in self.efa_links.values():
            for link in link_list:
                link.reset()

    # ----- Rank/device mapping -----

    def rank_to_node(self, rank):
        return rank // self.ranks_per_node

    def rank_to_local_rank(self, rank):
        return rank % self.ranks_per_node

    def rank_to_device(self, rank):
        """Global device index."""
        node = self.rank_to_node(rank)
        local_dev = self.rank_to_local_rank(rank) // self.cores_per_device
        return node * self.devices_per_node + local_dev

    def rank_to_local_device(self, rank):
        """Device index within its node (0..15)."""
        return self.rank_to_local_rank(rank) // self.cores_per_device

    def device_to_ranks(self, global_device):
        node = global_device // self.devices_per_node
        local_dev = global_device % self.devices_per_node
        base = node * self.ranks_per_node + local_dev * self.cores_per_device
        return list(range(base, base + self.cores_per_device))

    def node_to_ranks(self, node_id):
        base = node_id * self.ranks_per_node
        return list(range(base, base + self.ranks_per_node))

    def is_same_device(self, rank_a, rank_b):
        return self.rank_to_device(rank_a) == self.rank_to_device(rank_b)

    def is_same_node(self, rank_a, rank_b):
        return self.rank_to_node(rank_a) == self.rank_to_node(rank_b)

    # ----- Path and hop queries -----

    def device_path(self, src_dev, dst_dev):
        return self._shortest_paths[src_dev][dst_dev]

    def device_hops(self, src_dev, dst_dev):
        return len(self._shortest_paths[src_dev][dst_dev]) - 1

    def rank_hops(self, src_rank, dst_rank):
        return self.device_hops(
            self.rank_to_device(src_rank),
            self.rank_to_device(dst_rank)
        )

    # ----- Simulation -----

    def send(self, src_rank, dst_rank, bytes_, start_time=0.0):
        """
        Simulate sending bytes from src_rank to dst_rank.
        Routes intra-node via NeuronLink torus, inter-node via EFA.
        """
        src_node = self.rank_to_node(src_rank)
        dst_node = self.rank_to_node(dst_rank)

        if src_node == dst_node:
            # Intra-node: delegate to per-node topology
            local_src = self.rank_to_local_rank(src_rank)
            local_dst = self.rank_to_local_rank(dst_rank)
            return self.nodes[src_node].send(
                local_src, local_dst, bytes_, start_time)
        else:
            # Inter-node: route through EFA adapter
            src_dev_local = self.rank_to_local_device(src_rank)
            adapter_idx = src_dev_local % self.efa_adapters
            link = self.efa_links[(src_node, dst_node)][adapter_idx]
            return link.transmit(bytes_, start_time)

    # ----- Matrices and queries -----

    def hop_matrix(self):
        matrix = []
        for i in range(self.num_cores):
            row = []
            for j in range(self.num_cores):
                row.append(self.rank_hops(i, j))
            matrix.append(row)
        return matrix

    def neighbor_ranks(self, rank):
        """Return ranks that are 1 hop away (including same-device peer)."""
        node = self.rank_to_node(rank)
        local_rank = self.rank_to_local_rank(rank)
        base = node * self.ranks_per_node

        result = []
        # Same-device peers (all cores on same device except self)
        local_dev = local_rank // self.cores_per_device
        for c in range(self.cores_per_device):
            peer_local = local_dev * self.cores_per_device + c
            if peer_local != local_rank:
                result.append(base + peer_local)
        # NeuronLink neighbors (within same node)
        for nbr_dev in self.adjacency[local_dev]:
            for c in range(self.cores_per_device):
                result.append(base + nbr_dev * self.cores_per_device + c)
        return result

    def summary(self):
        hops = self.hop_matrix()
        max_hop = max(max(row) for row in hops)
        avg_hop = sum(sum(row) for row in hops) / (self.num_cores * self.num_cores)
        print(f"Trainium cluster topology ({self.num_nodes} node(s)):")
        print(f"  Nodes: {self.num_nodes}, Devices: {self.num_devices}, "
              f"Cores: {self.num_cores}")
        print(f"  NeuronLink BW: {self.link_bw} GB/s, "
              f"Latency: {self.link_lat} us")
        if self.num_nodes > 1:
            print(f"  EFA BW: {self.efa_bw} GB/s x {self.efa_adapters} "
                  f"adapters/node, Latency: {self.efa_lat} us")
        print(f"  Max hops: {max_hop}, Avg hops: {avg_hop:.2f}")
