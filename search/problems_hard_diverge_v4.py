"""VERY HARD divergence problems, batch v4 -- amplify the ONE real lever found.

v1-v3 result: under a fair (identical) gate, SorcarCCL (kiss) and OverlayCCL
(strat) TIE on every problem whose optimum is a clean structural idea or an
algebraic fold -- Sonnet 4.5 one-shots them in OverlayCCL's enumerate step.

The ONE place they diverged STRUCTURALLY was HD8 (blockwise_rs_bcast): the
baseline is framed as "all_reduce, slice my shard, all_gather the shards
back". OverlayCCL's enumerate step proposed FIVE strategies, ALL of which
preserved the shard/reassemble structure (AR+AG, RS+AG, all-to-all,
blockwise, hierarchical). None proposed the real optimum: DELETE the whole
shard round-trip and return a single all_reduce. kiss, iterating openly,
saw gathered == full and collapsed it -> 1 AR. But on HD8 the payoff was
only 1.039x because the simulator floors small collectives at ~5160us, so
the eliminated all_gather cost just ~200us -- below the 1.05 divergence bar.

v4 amplifies that exact lever: same "shard round-trip that is secretly a
no-op" structure, but with a LARGE payload so the redundant collective is
bandwidth-bound (cost >> floor). If the mechanism is real, the ratio should
now cross 1.05 under the identical gate. Each problem's optimum is a single
all_reduce; the baseline wraps it in a redundant, structure-priming
scatter/gather (or gather/reduce) round-trip that enumerate-once tends to
preserve.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _reg(name, sig_args, doc, ref_fn, gen_fn, builtin_code, call_args=None):
    sig = (f"def {name}_fn({sig_args}, rank, world_size, num_devices,\n"
           f"                 cores_per_device, xm, torch, num_nodes=1):")

    def _call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
        vals = [args[a] for a in (call_args or [sig_args.split(",")[0].strip()])]
        return fn(*vals, r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

    register_problem(CollectiveProblem(
        name=name, display_name=name, evolved_fn_name=f"{name}_fn",
        signature=sig, signature_doc=doc, reference_fn=ref_fn,
        generate_test_case=gen_fn, call_candidate=_call,
        builtin_templates={name: builtin_code}))


# ---------------------------------------------------------------------------
# HD14 = HD8 with a LARGE shard so the redundant all_gather is bandwidth-bound.
# Baseline: full AR (W*S,), slice this rank's S-shard, all_gather shards back.
# Net == AR(x). Optimum: single AR (drop the scatter/gather round-trip).
# S is large so the all_gather moves real bytes -> cost >> collective floor.
# ---------------------------------------------------------------------------
def _mk_big_rs_bcast(name, S=65536):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*S,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * S
        pra = [{'x': torch.randn(N) * (0.25 + 0.03 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}",
            "    # Full AR, keep only this rank's shard, then all_gather the",
            "    # shards back into the full (W*S,) vector.",
            "    full = xm.all_reduce(xm.REDUCE_SUM, x)   # (W*S,)",
            "    shard = full[rank*S:(rank+1)*S]          # (S,)",
            "    gathered = xm.all_gather(shard, dim=0)   # (W*S,)",
            "    return gathered"]
    _reg(name, "x", f"Local x ({'world*S'},), S={S} (large). Baseline: full "
         f"AR, slice this rank's S-shard, all_gather shards back to full. "
         f"Result = AR(x).",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD15 (gather-then-reduce round-trip, large): all_gather the full per-rank
# buffer, sum the gathered copies locally, then... that's already the answer,
# but the baseline ALSO all_reduces the result (a redundant second collective
# over a large payload). Optimum: single AR (or single AG + local sum), NOT
# both. The AG framing primes "gather then combine"; enumerate-once tends to
# keep the gather and just tweak the local combine, missing that the trailing
# AR is fully redundant. Large payload -> the redundant collective is costly.
# Baseline: 1 AG (W*S,)->(W*W*S,)... too big. Use per-rank (S,) -> AG (W*S,)
# then redundant AR (W*S,). Net == W * AR-mean... make it exact:
# baseline returns AR(SUM, all_gather(x).sum-fold) but structured redundantly.
# Simpler: y = all_gather(x) gives (W*S,) = concat of ranks; then baseline
# does AR(SUM, y) which sums the SAME concat across ranks = W * (concat).
# Optimum: W * all_gather(x). One collective instead of two.
# ---------------------------------------------------------------------------
def _mk_gather_then_redundant_reduce(name, S=65536):
    def _ref(inputs, world_size):
        # all_gather(x) on each rank = concat_r x_r  (same on all ranks) = (W*S,)
        concat = torch.cat([inp['x'] for inp in inputs])  # (W*S,)
        # AR(SUM) of that identical concat across W ranks = W * concat
        out = world_size * concat
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(S) * (0.25 + 0.03 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # Gather every rank's (S,) buffer into a (W*S,) concat, then",
            "    # all_reduce that concat (redundant: it's identical on all",
            "    # ranks, so AR just multiplies by world_size).",
            "    concat = xm.all_gather(x, dim=0)          # (W*S,)",
            "    reduced = xm.all_reduce(xm.REDUCE_SUM, concat)  # W * concat",
            "    return reduced"]
    _reg(name, "x", f"Local x ({S},) (large). Baseline: all_gather to a "
         f"(world*S,) concat, then all_reduce that concat (redundant). "
         f"Result = world * concat.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


def register_all():
    _mk_big_rs_bcast("hd14_big_rs_bcast")
    _mk_gather_then_redundant_reduce("hd15_gather_redundant_reduce")


register_all()
