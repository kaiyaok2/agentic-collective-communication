"""Round 3 -- framing-distance lever (attack overlay's enumerate-from-baseline).

Lesson L3: error-prone folds are seed-noise. Different structural attack:
OverlayCCL enumerates K=5 strategies by reasoning FROM THE BASELINE FRAMING.
If the true optimum lives in a structurally DISTANT framing (different collective
primitive, different data layout) that the baseline gives no hint toward, all K
enumerated strategies may stay in the baseline's neighborhood and miss it, while
kiss's open ReAct (which reframes freely across 30 steps) can migrate.

DIR-F "primitive swap": baseline expresses the computation with one collective
(e.g. all_gather + local reduce, O(W*N) bytes moved) but the optimum uses a
DIFFERENT primitive (reduce_scatter or all_reduce) that the all_gather framing
doesn't suggest. The all_gather-then-sum pattern is a common baseline habit;
recognizing it equals all_reduce(SUM) is a reframe, not a refinement.

  f1: sum over ranks via all_gather+sum  -> optimum all_reduce(SUM)
  f2: per-shard mean via all_gather+slice+mean -> optimum reduce_scatter/W then AG
  f3: max over ranks via all_gather+max  -> optimum all_reduce(MAX)

These are deterministic and gate-clean in BOTH the baseline and optimum forms
(no error-prone math), so any divergence is attributable to framing distance,
not to one side failing a cold implementation. If they TIE, it is strong
evidence that Sonnet reframes primitives just as well inside overlay's
enumerate step as inside kiss's loop -> parity is structural, not luck.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _reg(name, doc, ref_fn, gen_fn, builtin_code):
    sig = (f"def {name}_fn(x, rank, world_size, num_devices,\n"
           f"                 cores_per_device, xm, torch, num_nodes=1):")

    def _call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
        return fn(args["x"], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

    register_problem(CollectiveProblem(
        name=name, display_name=name, evolved_fn_name=f"{name}_fn",
        signature=sig, signature_doc=doc, reference_fn=ref_fn,
        generate_test_case=gen_fn, call_candidate=_call,
        builtin_templates={name: builtin_code}))


def _gen_flat(world_size, N, seed, scale=0.3):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (scale + 0.02 * r)} for r in range(world_size)]


# --- f1: all_gather+sum framing; optimum all_reduce(SUM) ---
def _mk_ag_sum(name, N=1024):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    # gather every rank's vector then sum locally (O(W*N) moved)",
            "    g = xm.all_gather(x, dim=0)          # (W*N,)",
            "    acc = g[0:N].clone()",
            "    for r in range(1, W):",
            "        acc = acc + g[r*N:(r+1)*N]",
            "    return acc"]
    _reg(name, f"Local x ({N},). Return the all-rank SUM. Baseline all_gathers "
         f"all vectors and sums them locally.", _ref, _gen, "\n".join(body) + "\n")


# --- f2: per-shard mean via all_gather+slice; optimum RS/W + AG ---
def _mk_ag_mean(name, N=1024):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs) / world_size
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    g = xm.all_gather(x, dim=0)",
            "    acc = g[0:N].clone()",
            "    for r in range(1, W):",
            "        acc = acc + g[r*N:(r+1)*N]",
            "    return acc / W"]
    _reg(name, f"Local x ({N},). Return the all-rank MEAN. Baseline all_gathers "
         f"then averages locally.", _ref, _gen, "\n".join(body) + "\n")


# --- f3: all_gather+max framing; optimum all_reduce(MAX) ---
def _mk_ag_max(name, N=1024):
    def _ref(inputs, world_size):
        st = inputs[0]['x'].clone()
        for inp in inputs[1:]:
            st = torch.maximum(st, inp['x'])
        return [st.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    g = xm.all_gather(x, dim=0)",
            "    rows = [g[r*N:(r+1)*N] for r in range(W)]",
            "    stacked = torch.stack(rows, dim=0)   # (W, N)",
            "    vals, _ = torch.max(stacked, dim=0)  # elementwise max over ranks",
            "    return vals"]
    _reg(name, f"Local x ({N},). Return the elementwise all-rank MAX. Baseline "
         f"all_gathers then reduces with maximum locally.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_ag_sum("r3_agsum_n1024", 1024)
    _mk_ag_sum("r3_agsum_n4096", 4096)
    _mk_ag_mean("r3_agmean_n1024", 1024)
    _mk_ag_mean("r3_agmean_n4096", 4096)
    _mk_ag_max("r3_agmax_n1024", 1024)
    _mk_ag_max("r3_agmax_n4096", 4096)


register_all()
