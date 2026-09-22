"""Round 13 -- REVERSE-DIRECTION HUNT + count-minimal CONTROLS.

Every confirmed divergence so far is Sorcar>Overlay, driven by kiss collapsing a
deep dependent-collective framing that overlay's enumerate-from-baseline + bounded
R=3 refinement stays trapped in. A defensible "one system dominates" conclusion
REQUIRES a genuine attempt at the reverse (Overlay>Sorcar). This round is that
attempt, plus controls that pin down WHY the reverse is hard under this sim.

Hypothesized reverse levers (Overlay could win if...):
DIR-U "over-engineering trap": baseline is ALREADY collective-minimal (1 AR).
The docstring dangles a plausible-but-useless "hierarchical two-level reduce"
optimization. Overlay enumerates from baseline and, finding no cheaper correct
form, keeps 1 AR. Kiss's open ReAct might chase the hint, introduce an extra
collective, and (if that scores no worse on a tie-broken seed) return a 2-collective
form. If kiss EVER regresses below overlay here, that is a real reverse signal.

DIR-V "unsound-fusion trap": baseline runs a genuine reduction that CANNOT be
count-reduced (already 1 collective, no redundancy). Kiss's freedom to reframe
is pure downside here -- any change is either a no-op or wrong. Tests whether
kiss's iteration ever lands on a slower-but-passing form.

CONTROL DIR-W "heterogeneous irreducible ops": baseline runs THREE different
reduce ops (SUM, MAX, MIN) on the same vector -- genuinely independent, cannot be
merged into one collective (different reduction semantics). Optimum = 3 collectives
for BOTH. Predicts a clean TIE. This isolates the claim: Sorcar's advantage exists
ONLY when the baseline has removable collective COUNT; when the count is already
irreducible, the two systems are equivalent. A tie here + Sorcar wins on deep
chains = the sharp, honest characterization of the divergence.
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


def _gen_flat(world_size, N, seed):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# --- DIR-U: already-minimal baseline, dangled useless "hierarchical" hint ---
def _mk_already_minimal(name, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # Already minimal: a single all_reduce is the whole computation.",
            "    return xm.all_reduce(xm.REDUCE_SUM, x)"]
    _reg(name, f"Local x ({N},). Return all-rank SUM. A single all_reduce suffices. "
         f"NOTE: some deployments prefer a two-level (intra-then-inter-node) "
         f"hierarchical reduce for large clusters; consider whether that helps here.",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-V: irreducible single reduction, no redundancy to exploit ---
def _mk_irreducible_single(name, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    return xm.all_reduce(xm.REDUCE_SUM, x)"]
    _reg(name, f"Local x ({N},). Return the element-wise SUM across all ranks.",
         _ref, _gen, "\n".join(body) + "\n")


# --- CONTROL DIR-W: three heterogeneous irreducible reduce ops ---
def _mk_hetero_ops(name, N=512):
    def _ref(inputs, world_size):
        stk = torch.stack([inp['x'] for inp in inputs], dim=0)  # (W, N)
        ssum = stk.sum(dim=0)
        smax = stk.max(dim=0).values
        smin = stk.min(dim=0).values
        out = ssum + smax + smin
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # Three DIFFERENT reduce ops on the same vector -- cannot be merged",
            "    # into one collective (distinct reduction semantics).",
            "    a = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    b = xm.all_reduce(xm.REDUCE_MAX, x)",
            "    c = xm.all_reduce(xm.REDUCE_MIN, x)",
            "    return a + b + c"]
    _reg(name, f"Local x ({N},). Return (all-rank SUM) + (all-rank MAX) + "
         f"(all-rank MIN), element-wise. Three distinct reductions.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_already_minimal("r13_minimal", 512)
    _mk_already_minimal("r13_minimal_big", 4096)
    _mk_irreducible_single("r13_single", 512)
    _mk_hetero_ops("r13_hetero", 512)


register_all()
