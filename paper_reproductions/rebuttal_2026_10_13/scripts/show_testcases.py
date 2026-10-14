import sys, torch; sys.path.insert(0, "/home/ubuntu/OverlayCCL")
from search.problems import get_problem
import search.problems_modext
def desc(v, depth=0):
    if isinstance(v, torch.Tensor): return f"T{tuple(v.shape)}:{str(v.dtype).replace('torch.','')}"
    if isinstance(v, (list, tuple)):
        if not v: return "[]"
        inner = desc(v[0], depth+1); return f"[{len(v)}x {inner}]"
    if isinstance(v, dict): return "{" + ", ".join(f"{k}={desc(x, depth+1)}" for k, x in list(v.items())[:8]) + "}"
    return repr(v)[:40]
for p in ("alltoallv","uniform_a2a","ring_kv","dxe","pp_send_recv","tp_mlp","fsdp_prefetch","llama_block_ar"):
    prob = get_problem(p)
    try:
        tc = prob.generate_test_case(224, "uniform", 64, seed=99)
        print(f"## {p}: per_rank_args[0] = {desc(tc['per_rank_args'][0])}\n     shared = {desc(tc['shared_args'])}")
    except Exception as e: print(f"## {p}: ERR {e}")
    import inspect; print("     gen:", inspect.signature(prob.generate_test_case), "| call:", inspect.signature(prob.call_candidate))
