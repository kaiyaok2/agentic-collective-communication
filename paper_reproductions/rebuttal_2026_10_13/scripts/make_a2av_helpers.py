"""Build per-primitive-harness AllToAllV candidates from the seed library: runtime files whose uniform helper
alltoallv(x, ws, mc) routes through evolved_alltoallv = <seed template>. Run ON master."""
import json, re, sys, os
sys.path.insert(0, "/home/ubuntu/OverlayCCL"); sys.path.insert(0, "/home/ubuntu/ap7")
from search.problems import get_problem
import search.problems_modext
from score_v3 import VEC
prob = get_problem("alltoallv")
base = open("/home/ubuntu/OverlayCCL/runtime/trainium_alltoallv_7node.py.paper.bak").read()
m = re.search(r'^def evolved_alltoallv\(.*?(?=^# =+\n# Public API)', base, re.S | re.M); assert m, "evolved fn block not found"
helper_old = re.search(r'^def alltoallv\(x, world_size, max_chunk\):.*', base, re.S | re.M); assert helper_old
HELPER = '''def alltoallv(x, world_size, max_chunk):
    """Uniform pre-packed AllToAllV routed through the agent path (seed-library candidate)."""
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")
    ws = world_size; mc = max_chunk
    return evolved_alltoallv(x, [mc] * ws, [mc] * ws, mc, _rank, ws, _NUM_DEVICES, _CORES_PER_DEVICE, xm, torch, num_nodes=_NUM_NODES)
'''
os.makedirs("/home/ubuntu/ap7/cands_a2av", exist_ok=True); man = []
srcs = dict(prob.builtin_templates); srcs["vectorized_view_col"] = VEC
for name, code in srcs.items():
    if name == "permute_ring": continue            # collective_permute: aborts/deadlocks on this SDK (archived + today)
    out = base[:m.start()] + code.rstrip() + "\n\n" + base[m.end():helper_old.start()] + HELPER
    p = f"/home/ubuntu/ap7/cands_a2av/alltoallv__tmpl_{name}.py"; open(p, "w").write(out)
    man.append({"problem": "alltoallv", "cand": f"tmpl_{name}", "kind": "file", "path": p})
json.dump(man, open("/home/ubuntu/ap7/manifest_a2av.json", "w"), indent=1)
print([c["cand"] for c in man])
