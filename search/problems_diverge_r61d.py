"""Round 61d -- FAMILY-3 TOP-UP #3: DATA-DEPENDENT diagonal collapse, MEANSQ g-factor sweep.

r61c confirmed exactly one variant -- meansq3 at payload p512 (best=1.16, ci=[1.037,1.0389]) --
while meansq3_p768 and the absmean variants missed the strict best-of-N. The signal: the
mean-of-squares factor at the SMALL p512 payload clears the gate. This battery presses that
proven regime: sweep the meansq coefficient at the winning p512 payload, plus meansq3 at two
new small payloads. Same data-dependent diagonal mechanism/telescoping as r47/r61/r61c
(each summed block b multiplied by 1 + c*mean(block^2); redundant AR per stage; per-block
inverse divides by the factor). MockTorch-traceable (only .mean()/arith). All (coeff, payload)
combos md5-distinct from registered (prescreen guard); depth is NOT distinctness.
"""
import torch  # noqa: F401
from .problems import CollectiveProblem, register_problem

NBLOCK = 8


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


def _gen(world_size, seed, part, nblock=NBLOCK):
    torch.manual_seed(seed)
    N = nblock * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _dd_code(name, part, depth, coeff):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}; C = {coeff}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    f = []",
              "    for b in range(B):",
              "        sb = s[b*S:(b+1)*S]",
              "        f.append(1.0 + C*(sb*sb).mean())",
              "    buf = s.clone()",
              "    for b in range(B):",
              "        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    for b in range(B):",
                  "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])"]
        else:
            L += ["    acc = acc / world_size"]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _dd_ref(part, depth, coeff):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        f = [1.0 + coeff * float((s[b * part:(b + 1) * part] ** 2).mean())
             for b in range(NBLOCK)]
        out = s.clone()
        for b in range(NBLOCK):
            out[b * part:(b + 1) * part] = s[b * part:(b + 1) * part] * f[b]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk_dd(name, part, depth, coeff, cue=True):
    ref = _dd_ref(part, depth, coeff)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, then each summed block b "
           f"is multiplied by a data-dependent factor (1 + {coeff} times the mean of the "
           f"squared entries of that block).")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, coeff))


def register_all():
    # sweep meansq coefficient at the proven p512 payload (coeff 3 already confirmed at p512;
    # coeff 2 == registered r61b_dd_meansq2_p512_d8, excluded by prescreen md5 guard, not re-added)
    _mk_dd("r61d_dd_meansq4_p512_d8", 512, 8, 4.0)
    _mk_dd("r61d_dd_meansq5_p512_d8", 512, 8, 5.0)
    _mk_dd("r61d_dd_meansq6_p512_d8", 512, 8, 6.0)
    _mk_dd("r61d_dd_meansq8_p512_d8", 512, 8, 8.0)
    _mk_dd("r61d_dd_meansq10_p512_d8", 512, 8, 10.0)
    # meansq3 (the confirmed coefficient) at two new small payloads
    _mk_dd("r61d_dd_meansq3_p256_d8", 256, 8, 3.0)
    _mk_dd("r61d_dd_meansq3_p384_d8", 384, 8, 3.0)


register_all()
