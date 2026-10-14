import os, sys, json
os.environ.setdefault("TABLE", "/home/ubuntu/hwprobe/hw_probe_measured.json"); os.environ.setdefault("MANIFEST", "/home/ubuntu/ap7/manifest_all.json")
os.environ.setdefault("RESET", "fusion"); os.environ.setdefault("MULTS", '{"ring_kv":2,"alltoallv":32,"uniform_a2a":384}')
sys.path.insert(0, "/home/ubuntu/ap7"); import score_v4 as SV
from search.problems import get_problem
from score_v3 import code_for
want = [tuple(x.split("/")) for x in sys.argv[1:]]
for p, cname in want:
    c = next(x for x in SV.man if x["problem"] == p and x["cand"] == cname); prob = get_problem(p); code = code_for(c, prob)
    if p in SV.REAL: SV.REAL[p](prob)
    ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
    kw = dict(SV.KW); kw["training_scale_bytes_multiplier"] = SV.MULTS.get(p, SV.DTYPE_MULT.get(p, 1.0) if p in SV.MULT1 else 1.0)
    b = SV.ct.benchmark_xla_candidate_generic(prob, fn, SV.TOPO, SV.send, SV.WORLD, num_nodes=SV.NODES, unsupported_primitives=["all_to_all"], **kw)
    print(f"\n### {p}/{cname}: sim={b.get('sim_time_us'):.1f} local={b.get('local_cost_us',0):.0f} comp={b.get('compilation_amortized_us',0):.0f} glo={b.get('graph_launch_overhead_us',0):.0f} hbm={b.get('hbm_penalty_us',0):.0f} fusion_credit?")
    print("   coll:", b.get("coll_breakdown"))
    ob = b.get("op_breakdown", {}); print("   ops :", {k: (v['count'], round(v['total_us'])) for k, v in sorted(ob.items(), key=lambda kv: -kv[1]['total_us'])[:8]})
