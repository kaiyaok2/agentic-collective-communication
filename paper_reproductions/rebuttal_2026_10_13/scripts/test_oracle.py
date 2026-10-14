import sys, ast, torch
sys.path.insert(0, "/home/ubuntu/OverlayCCL"); sys.path.insert(0, "/home/ubuntu/ap7")
import rewriter_search as R
from search.problems import get_problem
prob = get_problem("ring_kv")
for name in ["per_chunk_per_part_allgather", "per_slot_allgather", "joint_cat_allgather"]:
    code = prob.builtin_templates[name]; fn = R.compile_fn(prob, code)
    print(name, "sandbox:", R.correct(prob, fn), "real:", R.real_oracle(prob, fn))
start = ast.parse(prob.builtin_templates["per_chunk_per_part_allgather"]).body[0]
cands = list(R.expand(start)); fn2 = R.compile_fn(prob, R.src(cands[0][1]))
print("hoisted real:", R.real_oracle(prob, fn2))
tc = prob.generate_test_case(8, "uniform", 32, seed=0)
print("expected type:", type(tc["expected"]), "rank0:", [tuple(e.shape) for e in tc["expected"][0]] if isinstance(tc["expected"][0], list) else tc["expected"][0].shape)
print("rank0 kv shapes:", [tuple(t.shape) for t in tc["per_rank_args"][0]["kv_chunks"]])
