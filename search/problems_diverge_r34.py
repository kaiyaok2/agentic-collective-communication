"""Round 34 -- DISTINCT-SEMIRING PROBE (the honest second-family test).

Critique of the r33 conclusion: "no second family / trap is multiplicative-specific"
was UNDER-POWERED and partly a conflation. r33_shear tied not because it is
"non-multiplicative" but because a pairwise shear is ORDER-DEPENDENT / non-commuting
(its inverse is awkward). The real untested question is:

  Is the trap specific to (SUM, x) with a multiplicative per-shard scale, or is it
  ANY reversible per-shard op that COMMUTES THROUGH THE REDUCTION -- i.e. does it
  reproduce in a DIFFERENT SEMIRING with a DIFFERENT collective primitive?

Family-1 (all 19 prior robust wins): reduction = all_reduce(SUM), reversible per-block
op = MULTIPLICATIVE scale a[r]*block, which distributes through SUM -> a D-deep chain
collapses to 1 AR(SUM). This round builds the faithful TROPICAL analog:

  A. maxplus8 -- reduction = all_reduce(MAX), reversible per-block op = ADDITIVE offset
     b[r] + block. In the (max, +) semiring, adding a per-block constant COMMUTES through
     MAX exactly as scaling commutes through SUM: max_r(x[r] + b) = b + max_r(x[r]) when b
     is the same across the reduced ranks (it is -- b is a function of block index, applied
     identically on every rank). A D-deep chain of {add offset; AR(MAX); subtract offset}
     is therefore reducible to ONE AR(MAX) + a final offset. DIFFERENT primitive (MAX not
     SUM), DIFFERENT reversible op (+, not x), DIFFERENT algebra (tropical semiring).

  B. minplus8 -- the (min, +) dual: all_reduce(MIN) + additive offset. Same structure.

If maxplus/minplus CONFIRM at best-of-16 -> the trap is NOT multiplicative-specific; it
generalizes to any reduction with a commuting reversible per-shard op -> a genuine SECOND
family (distinct semiring + distinct primitive). If they TIE -> the SUM+multiplicative
specificity is real (and, since redundant AR(MAX) over already-reduced data is naturally
idempotent, a tie would say overlay sees through tropical idempotence the way it saw
through r10 re-max).

Offsets are AWKWARD (non-round: 0.3/0.5/0.7 per the r29 non-power-of-2 boundary) so overlay
can't one-shot the cancellation. Truthful count cue ("8 dependent all_reduce operations")
on the *_count8 variants (strongest trap per L11); *_res are result-only contrast.

fp32: the baseline is a genuine D=8 AR(MAX/MIN) chain (passes the gate; sim ~12k us). The
optimum is 1 AR(MAX/MIN) + final offset (~5160 us floor) -> ~2.4x headroom, same as family-1.
Correctness: after the first all_reduce every rank holds identical data, so each later
AR(MAX) over identical inputs is an exact identity (max of identical copies = the copy) --
NO normalization needed (unlike SUM, which needs /W). The add/subtract offset telescopes;
the final stage adds b without subtracting, so net = elementwise-reduce(inputs) + b.
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


def _gen_shards(world_size, seed, part):
    torch.manual_seed(seed)
    N = world_size * part
    # spread the per-rank means so the elementwise MAX/MIN across ranks is non-degenerate
    return [{'x': torch.randn(N) + 0.4 * r} for r in range(world_size)]


# ---------- tropical (MAX,+) / (MIN,+) telescope ----------
def _tropical_code(name, part, reduce_const, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    b = [0.3 + 0.2*(r % 3) for r in range(W)]",
         f"    s = xm.all_reduce(xm.{reduce_const}, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s.clone()",
              "    for r in range(W):",
              "        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] + b[r]",
              f"    s = xm.all_reduce(xm.{reduce_const}, buf)"]
        if not last:
            L += ["    for r in range(W):",
                  "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _tropical_ref(part, mode):
    def _ref(inputs, world_size):
        stacked = torch.stack([inp['x'] for inp in inputs], dim=0)
        red = stacked.max(dim=0).values if mode == "max" else stacked.min(dim=0).values
        out = red.clone()
        b = [0.3 + 0.2 * (r % 3) for r in range(world_size)]
        for r in range(world_size):
            out[r * part:(r + 1) * part] = red[r * part:(r + 1) * part] + b[r]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    max_ref = _tropical_ref(part, "max")
    min_ref = _tropical_ref(part, "min")

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    MAX_R = ("Final result = the elementwise MAX across ranks, with a per-block additive "
             "offset b[r]=0.3+0.2*(r%3) added to block r (a max-plus / tropical telescope).")
    MIN_R = ("Final result = the elementwise MIN across ranks, with a per-block additive "
             "offset b[r]=0.3+0.2*(r%3) added to block r (a min-plus / tropical telescope).")

    _reg("r34_maxplus8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{MAX_R}",
         max_ref, _mk_gen(max_ref), _tropical_code("r34_maxplus8_count8", part, "REDUCE_MAX"))
    _reg("r34_maxplus8_res",
         f"Local x (world*{part},), S={part}. {MAX_R}",
         max_ref, _mk_gen(max_ref), _tropical_code("r34_maxplus8_res", part, "REDUCE_MAX"))
    _reg("r34_minplus8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{MIN_R}",
         min_ref, _mk_gen(min_ref), _tropical_code("r34_minplus8_count8", part, "REDUCE_MIN"))
    _reg("r34_minplus8_res",
         f"Local x (world*{part},), S={part}. {MIN_R}",
         min_ref, _mk_gen(min_ref), _tropical_code("r34_minplus8_res", part, "REDUCE_MIN"))


register_all()
