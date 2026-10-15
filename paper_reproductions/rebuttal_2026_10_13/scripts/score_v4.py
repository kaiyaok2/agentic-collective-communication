#!/usr/bin/env python3
"""Arm V4 = measured probe table (in-graph marginal op costs) + the v4 simulator (correctness_test_v4.py:
per-rank share of node NIC bandwidth, bytes actually moved per collective, view ops at measured marginal cost,
elementwise/reduction ops charged by volume at measured HBM rate) + candidates traced at the per-problem
harness's REAL shapes for the payload-changing problems (dxe: T=256, V=32256; alltoallv: CAP*DM=1024 per pair)
with training-scale multiplier 1.0 there. Run ON master:
  cd /home/ubuntu/OverlayCCL && TABLE=/home/ubuntu/hwprobe/hw_probe_measured.json PYTHONPATH=. python score_v4.py
"""
import json, os, re, sys, importlib.util
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("ct_v4", os.environ.get("CT_V4", "/home/ubuntu/ap7/correctness_test_v4.py"))
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
if "concat" in os.environ.get("ABLATE", ""): ct._ABLATE_CONCAT = True
from search.problems import get_problem
import search.problems_modext
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts, _problem_train_scale_multiplier
import torch
WORLD, NODES = 224, 7
P = json.load(open(os.environ["TABLE"])); DT = P["derived_table"]; SP = P["sim_params"]
OPC = os.environ.get("OPC", "marginal")
op_costs = dict(DT["xla_op_overhead_us_marginal"] if OPC == "marginal" else DT["xla_op_overhead_us"])
if os.environ.get("LOCAL", "") == "snapshot":
    from search.agent_simulator_config import _HARDWARE_MEASUREMENTS as _HM
    op_costs = dict(_HM["xla_op_overhead_us"]); _mcs = _HM["memcpy_throughput"]
    SP = dict(SP, dispatch_overhead_us=100.0, graph_launch_overhead_us=_HM["graph_launch_overhead_us"]["per_mark_step_us"],
              memcpy_bytes_per_us=_mcs["strided_gbps"] * 1000.0, memcpy_seq_bytes_per_us=_mcs["sequential_gbps"] * 1000.0,
              pipeline_amort_alpha1=0.30, pipeline_amort_alpha2=0.10, pipeline_amort_alpha3=0.02)
    DT = dict(DT); DT["memcpy_throughput"] = dict(DT["memcpy_throughput"], clone_gbps=_mcs["sequential_gbps"]); print("LOCAL terms = snapshot")
KW = dict(op_costs=op_costs, dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"],
          compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000,
          pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
          memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"],
          hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"] * 1000.0,
          coll_curves=None, amort_knee_bytes=int(os.environ.get("KNEE", "262144")), local_fwd_bwd_factor=float(os.environ.get("FWDBWD", "1")), ablate=os.environ.get("ABLATE", ""),
          index_select_bytes_per_us=(1_000_000 * 4) / float(DT["index_select_scaling"]["large_index_1M"]) if "large_index_1M" in DT.get("index_select_scaling", {}) else 0.0,
          microbatch_loops_induce_graphs=(os.environ.get("SCOPE", "harness") == "e2e"),
          coll_run_reset=os.environ.get("RESET", "graph"))
TOPO = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5, num_devices_per_node=16, cores_per_device=2,
                         device_adjacency=None, efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
send = make_send_counts("moe", world=WORLD)
man = json.load(open(os.environ.get("MANIFEST", "/home/ubuntu/ap7/manifest_v2.json")))
def fit_curve(table, bytes_key_factor):
    """least-squares t = L + bytes_moved/BW over the probe's (bytes -> us) rows; bytes_moved = key bytes * factor"""
    pts = []
    for k, v in table.items():
        if not isinstance(v, (int, float)): continue
        nb = int(k.split("bytes_")[1].split("_")[0]); pts.append((nb * bytes_key_factor, float(v)))
    if len(pts) < 2: return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0; L = max(my - slope * mx, min(ys) * 0.5)
    bw = (1.0 / slope) if slope > 0 else 1e12
    return (round(L, 1), round(bw, 1))
CURVES = None
if os.environ.get("CURVES", "1") == "1":
    cws = P["world_size"]
    CURVES = {"all_gather": fit_curve(DT["all_gather_latency_us"], cws),              # key = input bytes; moved = ws*input (output)
              "all_reduce": fit_curve(DT["all_reduce_latency_us"], 2.0),              # ring: 2x bytes
              "reduce_scatter": fit_curve({k.replace("_in", ""): v for k, v in DT["reduce_scatter_latency_us"].items()}, 1.0)}
    CURVES = {k: v for k, v in CURVES.items() if v}
    # A-Q3 sensitivity arms: NET_SCALE scales every collective curve's bandwidth (and latency) -- a "different network";
    # LOCAL=snapshot swaps the LOCAL terms (op costs, memcpy, graph launch, dispatch) back to the shipped snapshot values.
    _ns = float(os.environ.get("NET_SCALE", "1"))
    if _ns != 1.0:
        CURVES = {k: (round(v[0] / _ns, 1), round(v[1] * _ns, 1)) for k, v in CURVES.items()}; print("NET_SCALE", _ns, CURVES)
    print("collective curves (L_us, bytes/us):", CURVES)
    KW["coll_curves"] = CURVES
if os.environ.get("TABLES", "1") == "1":
    T224 = os.environ.get("TABLE224")
    srcP = json.load(open(T224)) if T224 and os.path.exists(T224) else P
    DT2 = srcP["derived_table"] if "derived_table" in srcP else None
    sec = srcP["sections"]["collectives"]
    def tbl(d, strip=""):
        out = []
        for k, v in d.items():
            if isinstance(v, (int, float)): out.append((float(int(k.split("bytes_")[1].split("_")[0])), float(v)))
        return sorted(out)
    KW["coll_tables"] = {"all_gather": tbl(sec["all_gather"]), "all_reduce": tbl(sec["all_reduce"]), "reduce_scatter": tbl(sec["reduce_scatter"])}
    print("collective tables from", T224 or os.environ["TABLE"], {k: len(v) for k, v in KW["coll_tables"].items()}, "world", srcP["world_size"])
# ---- real-shape tracing for the payload-changing problems
def dxe_real(ws, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed); T, V = 256, 32256; V_local = V // ws
    targets = torch.randint(0, V_local * ws, (T,), dtype=torch.long)
    args = [{"logits_local": torch.randn(T, V_local), "targets": targets, "V_local": V_local} for r in range(ws)]
    try:
        from search.problems import _dxe_reference
        expected = _dxe_reference(args, ws)        # needed by the correctness oracle (generic search); cheap at ws<=8
    except Exception:
        expected = None
    return {"per_rank_args": args, "shared_args": {}, "expected": expected}
REAL = {"dxe": lambda prob: setattr(prob, "generate_test_case", dxe_real),
        "alltoallv": lambda prob: setattr(prob, "generate_test_case", (lambda g: (lambda ws, pattern="uniform", shard_size=64, seed=0: g(ws, pattern, 1024, seed)))(prob.generate_test_case))}
# real per-problem-harness shapes for the Llama-side problems (the generators take the shape kwargs directly);
# generators emit float32 while the harnesses run bf16 -> byte multiplier 0.5
if os.environ.get("REALSHAPES", "1") == "1" and "realshapes" not in os.environ.get("ABLATE", ""):
    def _wrap(g, **kw):
        return lambda ws, pattern="uniform", *a, **k: g(ws, pattern, **kw)
    REAL.update({
        "tp_mlp":         lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, N_LAYERS=1, B=1, S=1024, DM=2048, seed=0)),
        "fsdp_prefetch":  lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, N_LAYERS=1, shard_size=5376 // 224, DM=2048, seed=0)),
        "pp_send_recv":   lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, B=1, S=512, DM=2048, seed=0)),
        "llama_block_ar": lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, B=1, S=1024, DM=2048, seed=0)),
    })
MULT1 = set(REAL)
DTYPE_MULT = {"tp_mlp": 0.5, "fsdp_prefetch": 0.5, "pp_send_recv": 0.5, "llama_block_ar": 0.5}
# harness-faithful training-scale multipliers (bytes of the real per-problem harness tensor / bytes of the sandbox tensor);
# env MULTS=json overrides; default = the pipeline's own table except for the real-shape-traced problems
MULTS = json.loads(os.environ.get("MULTS", "{}"))
TAG = os.environ.get("TAG", OPC)
from score_v3 import code_for   # reuse the manifest -> source mapping (score_v3 import also scores arms A-C; harmless)
ONLY = [x for x in os.environ.get("PROBLEMS", "").split(",") if x]
def main():
  out = {}
  for c in man:
      if ONLY and c["problem"] not in ONLY: continue
      key = f"{c['problem']}__{c['cand']}"
      try:
          prob = get_problem(c["problem"]); code = code_for(c, prob)
          if c["problem"] in REAL: REAL[c["problem"]](prob)
          ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
          kw = dict(KW); kw["training_scale_bytes_multiplier"] = MULTS.get(c["problem"], DTYPE_MULT.get(c["problem"], 1.0) if c["problem"] in MULT1 else _problem_train_scale_multiplier(prob))
          b = ct.benchmark_xla_candidate_generic(prob, fn, TOPO, send, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **kw)
          v = None if "error" in b else round(b["sim_time_us"], 1)
          extra = (f"coll={b.get('total_bytes')} local={b.get('local_cost_us', 0):.0f} steps={b.get('steps')} neff={b.get('n_neff_graphs')} comp={b.get('compilation_amortized_us', 0):.0f} glo={b.get('graph_launch_overhead_us', 0):.0f}" if v is not None else b.get("error", "")[:80])
      except Exception as e:
          v = None; extra = f"{type(e).__name__}: {str(e)[:80]}"
      out[key] = v; print(f"{key:58s} V4={v}  {extra}", flush=True)
  OUTDIR = os.environ.get("OUTDIR", "/home/ubuntu/sweep_v2"); os.makedirs(OUTDIR, exist_ok=True)
  json.dump(out, open(f"{OUTDIR}/sim_v4_{TAG}.json", "w"), indent=1); print("wrote sim_v4", TAG)
if __name__ == "__main__":
  main()
