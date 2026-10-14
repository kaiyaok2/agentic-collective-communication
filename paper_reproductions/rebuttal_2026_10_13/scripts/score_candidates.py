#!/usr/bin/env python3
"""Score every manifest candidate with the Phase-3 simulator under ONE fixed, hand-set
calibration (the pipeline defaults: dispatch 100us, amort alphas .30/.10/.02, no LLM-fitted
op costs) at world=224 / 7 nodes.  Run ON master (needs torch for search.problems imports):
    cd /home/ubuntu/OverlayCCL && PYTHONPATH=. $VENV/bin/python /home/ubuntu/ap7/score_candidates.py
Writes /home/ubuntu/sweep_results/sim_default.json : {"<problem>__<cand>": sim_time_us | null}
"""
import json, os, re, sys, traceback
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
from search.problems import get_problem
from search.correctness_test import benchmark_xla_candidate_generic
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts

WORLD, NODES = 224, 7
topo = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                         num_devices_per_node=16, cores_per_device=2, device_adjacency=None)
send_counts = make_send_counts("moe", world=WORLD)
man = json.load(open(os.environ.get("MANIFEST", "/home/ubuntu/ap7/manifest.json")))
out = {}

def fn_from_code(problem, code):
    ns = {}; exec(code, ns); fn = ns[problem.evolved_fn_name]
    try: fn.__candidate_source__ = code
    except Exception: pass
    return fn

def evolved_from_runtime(path, fn_name):
    src = open(path).read()
    m = re.search(r'^def ' + fn_name + r'\(.*?(?=^def |^[A-Za-z_]\w* *=|\Z)', src, re.S | re.M)
    return m.group(0)

for c in man:
    key = f"{c['problem']}__{c['cand']}"
    try:
        prob = get_problem(c["problem"])
        if c["kind"] == "builtin":
            code = prob.builtin_templates[c["name"]]
        elif c["kind"] == "file":
            code = evolved_from_runtime(c["path"], prob.evolved_fn_name)
        elif c["kind"] == "code":
            code = open(c["code"]).read()
        else:  # harness baseline == the developer-baseline builtin, per problem
            base = {"alltoallv": "allgather_reduce_scatter", "uniform_a2a": "allgather_reduce_scatter",
                    "ring_kv": "per_chunk_per_part_allgather", "dxe": "full_vocab_ag",
                    "pp_send_recv": "per_mb", "tp_mlp": "per_mb", "fsdp_prefetch": "per_mb",
                    "llama_block_ar": "sequential_2ar"}[c["problem"]]
            code = prob.builtin_templates[base]
        fn = fn_from_code(prob, code)
        b = benchmark_xla_candidate_generic(prob, fn, topo, send_counts, WORLD, num_nodes=NODES,
                                            unsupported_primitives=["all_to_all"])
        out[key] = None if "error" in b else b["sim_time_us"]
        print(f"{key:60s} sim_us={out[key]}  {b.get('error','')[:60]}")
    except Exception as e:
        out[key] = None; print(f"{key:60s} EXC {str(e)[:100]}")
os.makedirs("/home/ubuntu/sweep_results", exist_ok=True)
outp = os.environ.get("SIM_OUT", "/home/ubuntu/sweep_results/sim_default.json"); json.dump(out, open(outp, "w"), indent=1)
print("wrote sim_default.json", len(out))
