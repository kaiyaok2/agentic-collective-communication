"""Probe: can the local mock gate validate a single-step collective_permute
problem? Ring shift: rank r must output rank (r-1)'s buffer.
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


def _mk_ring1(name, S=256):
    def _ref(inputs, world_size):
        # rank r receives rank (r-1)%W's buffer
        return [inputs[(r - 1) % world_size]['x'].clone()
                for r in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(S) * (1.0 + r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    pairs = [(s, (s+1) % world_size) for s in range(world_size)]",
            "    return xm.collective_permute(x, pairs)"]
    _reg(name, f"Ring shift, S={S}: rank r outputs rank (r-1)'s buffer via a "
         f"single collective_permute.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_ring1("ppb_ring_shift")


register_all()
