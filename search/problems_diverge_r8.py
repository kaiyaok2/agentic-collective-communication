"""Round 8 -- maximal framing seduction (informed by r1 mechanism finding).

r1 established the lever: overlay's enumerate-from-baseline-framing anchors it to
the baseline's multi-stage STRUCTURE, so it reaches the minimal-collective
optimum LESS reliably than kiss's open ReAct. The divergence widens when the
baseline framing is maximally seductive -- i.e. every stage looks individually
necessary and locally correct, but the whole thing collapses to ONE all_reduce
via a global algebraic identity that is NOT visible stage-by-stage.

DIR-N "telescoping chain": baseline computes a chain where stage i adds a term
that is later cancelled, so the net is a single sum. Each stage is a legit AR of
a plausible intermediate; only end-to-end algebra reveals the cancellation. The
baseline framing screams "sequential dependency"; the optimum is 1 AR.

DIR-O "per-stage-normalized accumulation": baseline reduces, normalizes by W,
scales, reduces again, re-normalizes ... N times. Each normalize/reduce pair is
individually sensible (looks like iterative averaging) but composes to a single
scaled sum. Overlay tends to keep the iterative structure and shave constants;
kiss collapses the recurrence.
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
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# --- DIR-N: telescoping chain (adds then cancels; net = one sum) ---
def _mk_telescope(name, part=256, nstage=4):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    # telescoping: t_k = AR(x) * c_k, accumulate with alternating",
            "    # signs whose coefficients sum to 1 => net == AR(x). Each stage",
            "    # is a real reduce of a plausible intermediate.",
            "    acc = None"]
    # coefficients c_k that sum to 1 (telescoping): 1 = (k+1)/n - k/n summed
    for k in range(nstage):
        # each stage reduces a scaled copy; scale chosen so total == 1*AR(x)
        num = k + 1
        body += [
            f"    r{k} = xm.all_reduce(xm.REDUCE_SUM, x * ({num}.0/{nstage}.0))",
            f"    r{k} = r{k} - xm.all_reduce(xm.REDUCE_SUM, x * ({k}.0/{nstage}.0))" if k > 0 else
            f"    # stage 0 contributes {num}/{nstage}",
        ]
        body += [
            f"    acc = r{k} if acc is None else acc + r{k}" if k == 0 else
            f"    acc = acc + r{k}",
        ]
    body += ["    return acc"]
    doc = (f"Local x (world*{part},). Compute all-rank SUM. Baseline uses a "
           f"{nstage}-stage telescoping accumulation (each stage reduces a "
           f"scaled copy; alternating contributions telescope to the plain sum).")
    _reg(name, doc, _ref, _gen, "\n".join(body) + "\n")


# --- DIR-O: iterative per-stage-normalized accumulation (recurrence) ---
def _mk_iter_norm(name, part=256, nstage=4):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        # net effect: scaled sum with per-shard coefficient a[r]
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    # iterative 'averaging then re-weighting': reduce, divide by W,",
            "    # multiply by W (identity), repeat; a final per-shard scale. The",
            "    # recurrence composes to a single scaled AR(x).",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for k in range(nstage - 1):
        body += [
            f"    s = xm.all_reduce(xm.REDUCE_SUM, s / W)   # iter {k}: reduce of replicated/W"]
    body += [
        "    out = s.clone()",
        "    for r in range(W):",
        "        out[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S]",
        "    return out"]
    doc = (f"Local x (world*{part},), S={part}. Baseline performs {nstage} "
           f"iterations of reduce-then-renormalize (iterative averaging) then a "
           f"per-shard re-weighting a[r]. Result = per-shard-scaled AR(x).")
    _reg(name, doc, _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_telescope("r8_telescope4", 256, nstage=4)
    _mk_telescope("r8_telescope6", 256, nstage=6)
    _mk_iter_norm("r8_iternorm4", 256, nstage=4)
    _mk_iter_norm("r8_iternorm6", 1024, nstage=6)


register_all()
