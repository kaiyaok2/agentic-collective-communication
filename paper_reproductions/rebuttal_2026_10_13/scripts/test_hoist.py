# Validate the R1 hoist on ring_kv with REAL torch semantics (fake xm: all_gather = cat over ranks)
import sys, ast, torch
sys.path.insert(0, "/home/ubuntu/OverlayCCL"); sys.path.insert(0, "/home/ubuntu/ap7")
import rewriter_search as R
from search.problems import get_problem
prob = get_problem("ring_kv"); code = prob.builtin_templates["per_chunk_per_part_allgather"]
start = ast.parse(code).body[0]
cands = list(R.expand(start)); print("expansions:", [c[0] for c in cands])
new_code = R.src(cands[0][1]); print(new_code)
WS = 4
class FakeXM:
    def __init__(self): self.calls = 0
    def all_gather(self, x, dim=0):
        self.calls += 1
        return torch.cat([x + 1000 * r for r in range(WS)], dim=dim)   # rank r's copy = x + 1000r
def run(c):
    ns = {}; exec(c, ns); fn = ns["evolved_ring_kv"]; xm = FakeXM()
    torch.manual_seed(0); kv = [torch.randn(64), torch.randn(64)]
    out = fn(kv, 0, WS, 2, 2, xm, torch)
    return [o.clone() for o in out], xm.calls
o1, c1 = run(code); o2, c2 = run(new_code)
print("calls original", c1, "hoisted", c2)
print("max diff per slot:", [float((a - b).abs().max()) for a, b in zip(o1, o2)], "shapes", [tuple(a.shape) for a in o1], [tuple(b.shape) for b in o2])
