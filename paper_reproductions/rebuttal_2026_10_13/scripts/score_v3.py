#!/usr/bin/env python3
"""Three-arm simulator scoring of the pool-v2 manifest, mirroring experiments/run_search.py's parameter
plumbing (A-Q2 correlation + A-Q3 calibration sensitivity):
  A  pipeline  : the pipeline's own calibration table (_HARDWARE_MEASUREMENTS: per-op costs ~29us, memcpy
                 200/20 GB/s, graph-launch 1500us, NEFF compile samples, amort alphas .3/.1/.02, per-problem
                 training-scale multiplier) — what the no-LLM / defaults path of run_search uses
  B  tpu-net   : arm A with the NETWORK constants swapped to TPU v5p values (ICI 600 GB/s, DCN 25 GB/s/port,
                 latencies), as in the paper's §5 swap
  C  no-opcost : arm A without the Phase-1-fitted local terms (per-op costs, memcpy, graph-launch, compile)
Run ON master: cd /home/ubuntu/OverlayCCL && PYTHONPATH=. $VENV/bin/python /home/ubuntu/ap7/score_v3.py
Writes /home/ubuntu/sweep_v2/sim_v3_{A,B,C}.json
"""
import json, os, re, sys
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
from search.problems import get_problem
import search.problems_modext
from search.correctness_test import benchmark_xla_candidate_generic
from search.agent_simulator_config import _HARDWARE_MEASUREMENTS as HM
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts, _problem_train_scale_multiplier

WORLD, NODES = 224, 7
man = json.load(open(os.environ.get("MANIFEST", "/home/ubuntu/ap7/manifest_v2.json")))
comp = HM.get("compilation_cost_us", {}) or {}
mc = HM.get("memcpy_throughput", {}) or {}
glo = (HM.get("graph_launch_overhead_us", {}) or {}).get("per_mark_step_us", 0.0)
OPC = dict(HM.get("xla_op_overhead_us", {}))

def topo(net):
    if net == "trn":
        return MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                                 num_devices_per_node=16, cores_per_device=2, device_adjacency=None,
                                 efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
    return MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=600.0, neuronlink_latency_us=1.0,
                             num_devices_per_node=16, cores_per_device=2, device_adjacency=None,
                             efa_bandwidth_GBps=25.0, efa_latency_us=2.0, efa_adapters_per_node=8)

ARMS = {
  "A": dict(topo=topo("trn"), kw=dict(op_costs=OPC, dispatch_overhead_us=100.0, graph_launch_overhead_us=glo,
            compilation_cost_samples=comp.get("samples", []), compilation_load_events_per_run=comp.get("load_events_per_run", 2),
            compilation_amortize_steps=comp.get("typical_training_steps", 5000),
            pipeline_amort_alpha1=0.30, pipeline_amort_alpha2=0.10, pipeline_amort_alpha3=0.02,
            memcpy_bytes_per_us=float(mc.get("strided_gbps", 0.0)) * 1000.0, memcpy_seq_bytes_per_us=float(mc.get("sequential_gbps", 0.0)) * 1000.0)),
  "C": dict(topo=topo("trn"), kw=dict(op_costs=None, dispatch_overhead_us=100.0, graph_launch_overhead_us=0.0,
            compilation_cost_samples=None, pipeline_amort_alpha1=0.30, pipeline_amort_alpha2=0.10, pipeline_amort_alpha3=0.02,
            memcpy_bytes_per_us=0.0, memcpy_seq_bytes_per_us=0.0)),
}
ARMS["B"] = dict(topo=topo("tpu"), kw=dict(ARMS["A"]["kw"]))
# M*: arms built from the REAL on-device probe table (train_hwprobe_7node.py -> hw_probe_measured.json):
#   M1 table-style op costs (isolated single-op cost for compute/copy ops, in-graph marginal for views)
#   M2 in-graph marginal op costs for every op
TABLE = os.environ.get("TABLE")
if TABLE:
    P = json.load(open(TABLE)); DT = P["derived_table"]; SP = P["sim_params"]
    def m_arm(opc):
        return dict(topo=topo("trn"), kw=dict(op_costs=opc, dispatch_overhead_us=SP["dispatch_overhead_us"],
                    graph_launch_overhead_us=SP["graph_launch_overhead_us"], compilation_cost_samples=SP["compilation_cost_samples"],
                    compilation_load_events_per_run=2, compilation_amortize_steps=5000,
                    pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
                    memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"]))
    ARMS["M1"] = m_arm(dict(DT["xla_op_overhead_us"])); ARMS["M2"] = m_arm(dict(DT["xla_op_overhead_us_marginal"]))
    for a in ("A", "B", "C"): ARMS.pop(a)
    print("probe-table arms:", {k: {kk: (vv if not isinstance(vv, (dict, list)) else "...") for kk, vv in v["kw"].items()} for k, v in ARMS.items()})
send = make_send_counts("moe", world=WORLD)

def evolved_from_runtime(path, fn_name):
    s = open(path).read()
    m = re.search(r'^def ' + fn_name + r'\(.*?(?=^def |^[A-Za-z_]\w* *=|\Z)', s, re.S | re.M); return m.group(0)
# AllToAllV per-problem harness runs the fixed helper; score helper variants by the algorithm the helper implements
A2AV_ALG = {"helper_AGRS": "ag_t_rs_inline_match", "helper_loop_cat_paper": "ag_slice_cat",
            "helper_view_col_public": "__vectorized__", "harness_baseline": "ag_t_rs_inline_match"}
VEC = '''def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    pack_size = world_size * max_chunk
    packed = torch.zeros(pack_size, device=input_tensor.device, dtype=input_tensor.dtype)
    off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = input_tensor[off:off + sc]
        off += sc
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(world_size, world_size, max_chunk)
    col = gathered[:, rank, :].contiguous()
    return torch.cat([col[s, :recv_counts[s]] for s in range(world_size)])
'''
BASE = {"alltoallv": "allgather_reduce_scatter", "uniform_a2a": "allgather_reduce_scatter", "ring_kv": "per_chunk_per_part_allgather",
        "dxe": "full_vocab_ag", "pp_send_recv": "per_mb", "tp_mlp": "per_mb", "fsdp_prefetch": "per_mb", "llama_block_ar": "sequential_2ar"}

# the per-primitive harness runs the runtime's UNIFORM helper alltoallv(x, ws, mc) on a pre-packed buffer; score each helper
# variant by the code the helper actually executes (no re-pack loop), expressed in the evolved_alltoallv signature
HELPER_CODE = {
 "helper_view_col_public": '''def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0).view(world_size, world_size, max_chunk)
    my_col = gathered[:, rank, :].contiguous()
    return my_col.view(-1)
''',
 "helper_loop_cat_paper": '''def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0).view(world_size, -1)
    chunks = []
    for src in range(world_size):
        chunks.append(gathered[src, rank * max_chunk:rank * max_chunk + max_chunk])
    return torch.cat(chunks)
''',
}
def code_for(c, prob):
    if c["problem"] == "alltoallv" and c["cand"] in HELPER_CODE and os.environ.get("HELPER_CODE", "1") == "1":
        return HELPER_CODE[c["cand"]]
    if c["problem"] == "alltoallv" and c["cand"] in A2AV_ALG:
        alg = A2AV_ALG[c["cand"]]; return VEC if alg == "__vectorized__" else prob.builtin_templates[alg]
    if c["kind"] == "builtin": return prob.builtin_templates[c["name"]]
    if c["kind"] == "code": return open(c["code"]).read()
    if c["kind"] == "file": return evolved_from_runtime(c["path"], prob.evolved_fn_name)
    return prob.builtin_templates[BASE[c["problem"]]]

out = {a: {} for a in ARMS}
for c in man:
    key = f"{c['problem']}__{c['cand']}"
    try:
        prob = get_problem(c["problem"]); code = code_for(c, prob)
        ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
    except Exception as e:
        for a in ARMS: out[a][key] = None
        print(f"{key:58s} LOADFAIL {str(e)[:80]}"); continue
    row = []
    for a, spec in ARMS.items():
        try:
            kw = dict(spec["kw"]); kw["training_scale_bytes_multiplier"] = _problem_train_scale_multiplier(prob)
            b = benchmark_xla_candidate_generic(prob, fn, spec["topo"], send, WORLD, num_nodes=NODES,
                                                unsupported_primitives=["all_to_all"], **kw)
            v = None if "error" in b else round(b["sim_time_us"], 1)
        except Exception as e:
            v = None
        out[a][key] = v; row.append(f"{a}={v}")
    print(f"{key:58s} " + "  ".join(row), flush=True)
OUTDIR = os.environ.get("OUTDIR", "/home/ubuntu/sweep_v2"); os.makedirs(OUTDIR, exist_ok=True)
for a in ARMS: json.dump(out[a], open(f"{OUTDIR}/sim_v3_{a}.json", "w"), indent=1)
print("wrote sim_v3_A/B/C.json")
