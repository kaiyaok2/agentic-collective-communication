import json, os, sys
sys.path.insert(0, "/home/ubuntu/OverlayCCL"); sys.path.insert(0, "/home/ubuntu/ap7")
os.environ["TABLE"] = "/home/ubuntu/hwprobe/hw_probe_measured.json"; os.environ["MANIFEST"] = "/home/ubuntu/ap7/manifest_min.json"
json.dump([], open("/home/ubuntu/ap7/manifest_min.json", "w"))
import score_v3 as SV
from search.problems import get_problem
from search.correctness_test import benchmark_xla_candidate_generic
from experiments.run_search import _problem_train_scale_multiplier
man = json.load(open("/home/ubuntu/ap7/manifest_v2.json"))
want = [("dxe", "g_structure-one_ag"), ("dxe", "g_structure-three_ar"), ("dxe", "paper_e791408f"), ("alltoallv", "helper_loop_cat_paper"), ("alltoallv", "helper_view_col_public"), ("alltoallv", "helper_AGRS"), ("tp_mlp", "g_bundle-per_layer_pack-flat"), ("tp_mlp", "harness_baseline")]
for p, cname in want:
    c = next(x for x in man if x["problem"] == p and x["cand"] == cname); prob = get_problem(p); code = SV.code_for(c, prob)
    ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
    kw = dict(SV.ARMS["M1"]["kw"]); kw["training_scale_bytes_multiplier"] = _problem_train_scale_multiplier(prob)
    b = benchmark_xla_candidate_generic(prob, fn, SV.ARMS["M1"]["topo"], SV.send, SV.WORLD, num_nodes=SV.NODES, unsupported_primitives=["all_to_all"], **kw)
    print(f"\n### {p}/{cname}  train_scale_mult={kw['training_scale_bytes_multiplier']}")
    for k, v in b.items():
        if isinstance(v, (int, float)) or (isinstance(v, (dict, list)) and len(str(v)) < 400): print(f"   {k}: {v}")
