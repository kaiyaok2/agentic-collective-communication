"""Sketch of a simulator extension that scores model-extension compositions.

The current cost model (paper Eq. 1) is:
  T_step = T_local + T_coll + T_launch + T_NEFF + T_net
with T_launch = 2 * l * m (l = per-mark_step framework cost, m = number
of autograd.Function boundaries, factor 2 covers fwd + bwd mark_step).

This is correct for the four headline OLMoE problems because each
problem's composition lives in one autograd.Function and XLA fuses all
of its collectives inside one HLO graph.

The model-extension setting (PP, TP+microbatching, FSDP+microbatching,
all-three-composed) puts each microbatch's work in its own mark_step
graph, so T_launch becomes 2 * l * m * M and T_coll changes shape:
the M graphs each pay d_full once for their fused-collective bundle.

Below is the minimal extension. It introduces:

  microbatch_count    : int M (1 if no per-mb mark_step pattern)
  per_mb_collectives  : list of (primitive_type, bytes) per microbatch
                        (the same shape for all M μbatches; bundling
                         replaces M*K dispatches with K of size M*bytes).

The agent's win is then directly visible to the simulator as the
difference between the per_mb and bundled scoring of the same
problem definition.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MicrobatchedComposition:
    """A composition that is repeated M times with a per-mb mark_step.

    Two scoring modes:
      'per_mb'   : M separate dispatches, M*l mark_step cost
      'bundled'  : 1 dispatch with M-fold bytes, 1*l mark_step cost
    """
    # Existing CollectiveProblem fields applicable per microbatch...
    per_mb_local_ops: List[Tuple[str, int]]   # (op_name, bytes) per μbatch
    per_mb_collectives: List[Tuple[str, int]] # (primitive, bytes) per μbatch
    microbatch_count: int = 1                  # M
    bundling_mode: str = 'per_mb'              # 'per_mb' or 'bundled'


def score_microbatched(composition: MicrobatchedComposition, cost_constants):
    """Per-step cost for a microbatched composition.

    cost_constants: dict with keys 'd_full_fn'(primitive, bytes) -> us,
                                   'd_amort', 'l', 'C_fn'(bytes) -> us,
                                   'op_floor', 'seq_bw', 'strided_bw'.
    """
    M = composition.microbatch_count
    mode = composition.bundling_mode

    if mode == 'per_mb':
        # Each microbatch's collectives form ONE in-graph bundle that XLA
        # fuses (collectives within the same mark_step graph). M graphs
        # in total, each charges d_full once plus k-1 amortised, where k
        # is the number of collectives in one microbatch graph.
        k = len(composition.per_mb_collectives)
        if k == 0:
            T_coll_per_mb = 0.0
        else:
            largest = max(b for _, b in composition.per_mb_collectives)
            T_coll_per_mb = (cost_constants['d_full_fn']('all_reduce', largest)
                             + (k - 1) * cost_constants['d_amort'])
        T_coll = M * T_coll_per_mb

        # M separate mark_step graphs: pay launch overhead M times (fwd)
        # + M times (bwd) under the same 2*l*m logic as Eq. 1.
        T_launch = 2 * cost_constants['l'] * M

        # M separate NEFFs (one per microbatch graph).
        # Largest single-collective tensor is the same per microbatch
        # (M does not multiply payload in per_mb mode).
        largest_per_mb = max((b for _, b in composition.per_mb_collectives), default=0)
        T_NEFF_per_step = M * cost_constants['C_fn'](largest_per_mb)

    elif mode == 'bundled':
        # All M microbatches' work lives in ONE mark_step graph. Each
        # collective primitive type is dispatched once with M-fold bytes.
        # Group by primitive type:
        from collections import defaultdict
        type_bytes = defaultdict(int)
        for prim, bytes_ in composition.per_mb_collectives:
            type_bytes[prim] += M * bytes_
        # Each unique primitive type pays one d_full at the bundled size.
        T_coll = 0.0
        largest = 0
        for prim, total_bytes in type_bytes.items():
            T_coll += cost_constants['d_full_fn'](prim, total_bytes)
            largest = max(largest, total_bytes)
        # Subsequent dispatches inside this single graph amortise.
        if len(type_bytes) > 1:
            T_coll += (len(type_bytes) - 1) * cost_constants['d_amort']

        # One mark_step graph: pay launch overhead twice (fwd + bwd).
        T_launch = 2 * cost_constants['l']
        # One NEFF, sized by largest bundled collective.
        T_NEFF_per_step = cost_constants['C_fn'](largest)
    else:
        raise ValueError(mode)

    # Local ops scale linearly with M either way (we still do M
    # microbatches of compute regardless of mark_step placement).
    T_local = 0.0
    for op, bytes_ in composition.per_mb_local_ops:
        T_local += cost_constants['op_floor']   # simplistic; real cost
                                                 # would dispatch through
                                                 # the per-op table
    T_local *= M

    return T_local + T_coll + T_launch + T_NEFF_per_step


# Example: Llama-style 4-composition step at the shapes the cluster runs.
# These are illustrative — the real cost_constants come from Phase 1
# probe results, not hardcoded.
EXAMPLE_COST = {
    'd_full_fn': lambda prim, b: 1500 + b / (50e9 / 1e6),  # us
    'd_amort':   150,
    'l':         500,                                       # us per mark_step
    'C_fn':      lambda b: 200 + b / 1e7,                   # us
    'op_floor':  1,
    'seq_bw':    100e9,
    'strided_bw': 25e9,
}

if __name__ == '__main__':
    # Example: PP cross-stage send/recv at M=4
    pp = MicrobatchedComposition(
        per_mb_local_ops=[('matmul', 0)] * 4,
        per_mb_collectives=[('all_reduce', 224 * 1 * 512 * 2048 * 2)],  # 224MB buffer
        microbatch_count=4,
    )
    pp.bundling_mode = 'per_mb'
    per_mb_score = score_microbatched(pp, EXAMPLE_COST)
    pp.bundling_mode = 'bundled'
    bundled_score = score_microbatched(pp, EXAMPLE_COST)
    print(f"PP M=4 per_mb={per_mb_score:.0f}us bundled={bundled_score:.0f}us "
          f"speedup={per_mb_score/bundled_score:.2f}x")
    # Expected: speedup > 1 because per_mb pays 4*l + 4*d_full while
    # bundled pays 1*l + 1*d_full(4x bytes); 4*d_full > 1*d_full(4x) on
    # this Trainium cost surface.
