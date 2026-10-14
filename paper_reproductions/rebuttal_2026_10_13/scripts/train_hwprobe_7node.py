#!/usr/bin/env python3
"""Phase-1 hardware probes, REAL on-device version (rebuttal, 2026-10-13).

Measures every quantity the Phase-1 tools in search/agent_simulator_config.py report
(xla_op_overhead_us, memcpy_throughput, graph_launch_overhead_us, compilation_cost_us,
back_to_back_amortization_us, index_select_scaling, all_gather/reduce_scatter/all_reduce latency,
collective_permute latency) on the running cluster, and writes them in the _HARDWARE_MEASUREMENTS
schema plus the derived simulator parameters (op_costs, dispatch_overhead_us, pipeline alphas via the
tool handler's own fit, graph-launch, memcpy bytes/us, compile samples).

Launch like a harness (single node = 32 ranks, as Phase 1 did; COLD=1 so every graph's first execution
is a real compile+load):
  COLD=1 WORKERS_OVERRIDE=" " PERCALL_PROBE=0 RESULTS_DIR=/home/ubuntu/hwprobe \
    bash /home/ubuntu/ap7/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe
Rank 0 writes $RESULTS_DIR/hw_probe_measured.json.
"""
import argparse, json, os, statistics, time, traceback
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

p = argparse.ArgumentParser()
p.add_argument("--backend", default="probe"); p.add_argument("--steps", type=int, default=1); p.add_argument("--warmup", type=int, default=3)
p.add_argument("--iters", type=int, default=int(os.environ.get("PROBE_ITERS", "15")))
args = p.parse_args()

dist.init_process_group("xla")
dev = xm.xla_device()
rank = xr.global_ordinal(); ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
OUT = os.environ.get("RESULTS_DIR", "/home/ubuntu/hwprobe")
WARM, ITERS = args.warmup, args.iters
R = {"world_size": ws, "sections": {}}
def log(*a):
    if rank == 0: print(f"[probe {time.strftime('%H:%M:%S')}]", *a, flush=True)

def materialize(*ts):
    xm.mark_step(); xm.wait_device_ops(); return ts

def timeit(build, warm=WARM, iters=ITERS):
    """build() enqueues lazy ops and returns the live outputs. Returns first-exec (compile+load) and
    steady per-graph wall time (median/min over iters), in us."""
    t0 = time.perf_counter(); out = build(); xm.mark_step(); xm.wait_device_ops(); first = time.perf_counter() - t0
    for _ in range(warm):
        out = build(); xm.mark_step(); xm.wait_device_ops()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); out = build(); xm.mark_step(); xm.wait_device_ops(); ts.append(time.perf_counter() - t0)
    del out
    return dict(first_us=round(first * 1e6, 1), median_us=round(statistics.median(ts) * 1e6, 1), min_us=round(min(ts) * 1e6, 1))

def section(name, fn):
    log("==", name)
    try:
        R["sections"][name] = fn()
    except Exception as e:
        R["sections"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        if rank == 0: traceback.print_exc()
    xm.mark_step(); xm.wait_device_ops()
    if rank == 0:
        os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, "hw_probe_partial.json"), "w"), indent=1)

# ----------------------------------------------------------------------------- 0. empty / trivial graph
def s_trivial():
    x, = materialize(torch.randn(32768, device=dev))
    res = {"empty_mark_step": timeit(lambda: [])}
    res["add_1op"] = timeit(lambda: [x + 1.0])
    return res

# ----------------------------------------------------------------------------- 1. xla op overhead (isolated vs in-graph marginal)
def s_xla_ops():
    N = 32768; n_many = 16
    xs = [torch.randn(N, device=dev) for _ in range(n_many)]
    x2 = [torch.randn(N, device=dev) for _ in range(n_many)]
    idx = [torch.randperm(N).to(dev) for _ in range(n_many)]
    materialize(*xs, *x2, *idx)
    OPS = {
        "add": lambda i: xs[i] + 1.0,
        "index_select": lambda i: xs[i].index_select(0, idx[i]),
        "slice": lambda i: xs[i][1:],
        "cat": lambda i: torch.cat([xs[i], x2[i]]),
        "stack": lambda i: torch.stack([xs[i], x2[i]]),
        "zeros": lambda i: torch.zeros(N, device=dev) + xs[i][:1],
        "tensor": lambda i: torch.tensor([float(i)] * 8, device=dev),
        "gather": lambda i: xs[i].gather(0, idx[i]),
        "scatter": lambda i: torch.zeros_like(xs[i]).scatter(0, idx[i], x2[i]),
        "split": lambda i: xs[i].split(N // 2),
        "chunk": lambda i: xs[i].chunk(2),
        "view": lambda i: xs[i].view(2, -1),
        "reshape": lambda i: xs[i].reshape(2, -1),
        "unsqueeze": lambda i: xs[i].unsqueeze(0),
        "squeeze": lambda i: xs[i].unsqueeze(0).squeeze(0),
        "flatten": lambda i: xs[i].view(2, -1).flatten(),
        "narrow": lambda i: xs[i].narrow(0, 8, N - 16),
        "transpose": lambda i: xs[i].view(2, -1).transpose(0, 1),
        "permute": lambda i: xs[i].view(2, -1).permute(1, 0),
        "expand": lambda i: xs[i].unsqueeze(0).expand(2, N),
        "contiguous": lambda i: xs[i].view(2, -1).transpose(0, 1).contiguous(),
        # the two copies the simulator's memcpy terms model, at this small size
        "narrow_nonlead_contiguous": lambda i: xs[i].view(32, 32, -1)[:, 3, :].contiguous(),
        "permute_contiguous": lambda i: xs[i].view(32, 32, -1).permute(1, 0, 2).contiguous(),
    }
    def flat(outs):
        o = []
        for t in outs: o.extend(t if isinstance(t, (tuple, list)) else [t])
        return o
    res = {}
    for name, f in OPS.items():
        r1 = timeit(lambda: flat([f(0)])); r16 = timeit(lambda: flat([f(i) for i in range(n_many)]))
        res[name] = dict(isolated_1op_us=r1["median_us"], graph_16ops_us=r16["median_us"],
                         marginal_us=round((r16["median_us"] - r1["median_us"]) / (n_many - 1), 2), first_us=r1["first_us"])
        log(f"  {name:28s} 1op={r1['median_us']:8.1f}  16ops={r16['median_us']:8.1f}  marginal={res[name]['marginal_us']:7.2f}")
    return res

# ----------------------------------------------------------------------------- 2. index_select scaling
def s_index_select():
    src, = materialize(torch.randn(1 << 20, device=dev))
    res = {}
    for name, n in (("small_index_100", 100), ("medium_index_10k", 10_000), ("large_index_1M", 1_000_000)):
        idx, = materialize(torch.randint(0, 1 << 20, (n,)).to(dev))
        r = timeit(lambda: [src.index_select(0, idx)]); res[name] = r["median_us"]; log(f"  {name}: {r['median_us']} us")
    return res

# ----------------------------------------------------------------------------- 3. memcpy throughput (dense permute vs sub-region narrow)
def s_memcpy():
    res = {}
    for mc in (4096, 16384):           # (32,32,mc) fp32 = 16 MB / 64 MB
        X, = materialize(torch.randn(32, 32, mc, device=dev)); nbytes = X.numel() * 4
        clone = timeit(lambda: [X.clone()])
        perm = timeit(lambda: [X.permute(1, 0, 2).contiguous()])
        col = timeit(lambda: [X[:, rank % 32, :].contiguous()])
        res[f"tensor_{nbytes >> 20}MB"] = dict(
            bytes=nbytes,
            clone_us=clone["median_us"], clone_gbps=round(nbytes / clone["median_us"] / 1e3, 2),
            permute_contiguous_us=perm["median_us"], sequential_gbps=round(nbytes / perm["median_us"] / 1e3, 2),
            narrow_col_contiguous_us=col["median_us"],
            strided_gbps_full_tensor_basis=round(nbytes / col["median_us"] / 1e3, 2),
            strided_gbps_actual_bytes_basis=round((nbytes / 32) / col["median_us"] / 1e3, 2))
        log(f"  {nbytes>>20}MB: clone {clone['median_us']}us perm {perm['median_us']}us col {col['median_us']}us")
    return res

# ----------------------------------------------------------------------------- 4. graph-launch (mark_step) overhead
def s_graph_launch():
    x, = materialize(torch.randn(32768, device=dev))
    ts = [torch.randn(1024, device=dev) for _ in range(16)]; materialize(*ts)
    def k_steps_local(K):
        def b():
            outs = []
            for k in range(K):
                outs.append(x * 1.0001 + 0.5)
                if k < K - 1: xm.mark_step()
            return outs
        return b
    def k_steps_ar_separate(K):
        def b():
            outs = []
            for k in range(K):
                outs.append(xm.all_reduce(xm.REDUCE_SUM, ts[k]))
                if k < K - 1: xm.mark_step()
            return outs
        return b
    def k_ar_fused(K):
        return lambda: [xm.all_reduce(xm.REDUCE_SUM, ts[k]) for k in range(K)]
    res = {}
    for K in (1, 4, 16):
        a = timeit(k_steps_local(K)); b = timeit(k_steps_ar_separate(K)); c = timeit(k_ar_fused(K))
        res[f"K{K}"] = dict(local_K_mark_steps_us=a["median_us"], ar_K_mark_steps_us=b["median_us"], ar_K_in_one_graph_us=c["median_us"])
        log(f"  K={K}: local-sep {a['median_us']}  AR-sep {b['median_us']}  AR-fused {c['median_us']}")
    res["per_mark_step_us_local"] = round((res["K16"]["local_K_mark_steps_us"] - res["K1"]["local_K_mark_steps_us"]) / 15, 1)
    res["per_mark_step_us_collective"] = round((res["K16"]["ar_K_mark_steps_us"] - res["K16"]["ar_K_in_one_graph_us"]) / 15, 1)
    return res

# ----------------------------------------------------------------------------- 5. back-to-back amortization (N ARs in one graph)
def s_back_to_back():
    ts = [torch.randn(1024, device=dev) for _ in range(32)]; materialize(*ts)
    res = {}
    for d in (1, 2, 4, 8, 16, 32):
        r = timeit(lambda: [xm.all_reduce(xm.REDUCE_SUM, ts[i]) for i in range(d)]); res[f"depth_{d}_total_us"] = r["median_us"]
        ra = timeit(lambda: [xm.all_gather(ts[i], dim=0) for i in range(d)]); res[f"ag_depth_{d}_total_us"] = ra["median_us"]
        log(f"  depth {d}: AR {r['median_us']} us   AG {ra['median_us']} us")
    res["dispatch_overhead_us_first_issue"] = res["depth_1_total_us"]
    return res

# ----------------------------------------------------------------------------- 6. collective latency vs bytes (+ compile cost of the AG graph)
def s_collectives():
    res = {"all_gather": {}, "reduce_scatter": {}, "all_reduce": {}, "compile_first_exec_us": {}}
    SIZES = [int(x) for x in os.environ.get("SIZES", "4096,32768,65536,524288,4194304,8388608,12582912,50331648,100663296").split(",")]
    AR_ONLY = [int(x) for x in os.environ.get("AR_ONLY_SIZES", "").split(",") if x]
    for nbytes in AR_ONLY:          # large all_reduce points (AR output == input, so the compile stays cheap)
        try:
            x, = materialize(torch.randn(nbytes // 4, device=dev))
            r2 = timeit(lambda: [xm.all_reduce(xm.REDUCE_SUM, x)], iters=7); res["all_reduce"][f"world_{ws}_bytes_{nbytes}"] = r2["median_us"]
            log(f"  AR-only {nbytes}B: steady {r2['median_us']} us, first-exec {r2['first_us']/1e6:.1f} s"); del x
        except Exception as e:
            log(f"  AR-only {nbytes}B failed: {e}")
        xm.mark_step(); xm.wait_device_ops()
    for nbytes in SIZES:
        n = nbytes // 4
        try:
            x, = materialize(torch.randn(n, device=dev))
            r = timeit(lambda: [xm.all_gather(x, dim=0)], iters=7 if nbytes >= 50331648 else ITERS)
            res["all_gather"][f"world_{ws}_bytes_{nbytes}"] = r["median_us"]; res["compile_first_exec_us"][str(nbytes)] = r["first_us"]
            log(f"  AG {nbytes}B: steady {r['median_us']} us, first-exec {r['first_us']/1e6:.1f} s")
            if nbytes <= 8388608:
                r2 = timeit(lambda: [xm.all_reduce(xm.REDUCE_SUM, x)]); res["all_reduce"][f"world_{ws}_bytes_{nbytes}"] = r2["median_us"]
                xg, = materialize(torch.randn(n * ws, device=dev))
                r3 = timeit(lambda: [xm.reduce_scatter(xm.REDUCE_SUM, xg, scale=1.0, scatter_dim=0, shard_count=ws)])
                res["reduce_scatter"][f"world_{ws}_bytes_{nbytes * ws}_in"] = r3["median_us"]
                log(f"     AR {r2['median_us']} us   RS(in {nbytes*ws}B) {r3['median_us']} us")
            del x
        except Exception as e:
            res["compile_first_exec_us"][str(nbytes)] = f"ERR {type(e).__name__}: {str(e)[:120]}"; log(f"  {nbytes}B failed: {e}")
        xm.mark_step(); xm.wait_device_ops()
    return res

# ----------------------------------------------------------------------------- 7. collective_permute latency
def s_permute():
    x, = materialize(torch.randn(1024, device=dev)); res = {}
    for steps in (1, 15, 31):
        pairs = [(i, (i + steps) % ws) for i in range(ws)]
        try:
            r = timeit(lambda: [xm.collective_permute(x, pairs)]); res[f"{steps}_step_bytes_4096"] = r["median_us"]; log(f"  CP {steps} step: {r['median_us']} us")
        except Exception as e:
            res[f"{steps}_step_bytes_4096"] = f"ERR {str(e)[:100]}"
    return res

SECTIONS = [x for x in os.environ.get("SECTIONS", "").split(",") if x]
for name, fn in (("trivial", s_trivial), ("xla_ops", s_xla_ops), ("index_select", s_index_select), ("memcpy", s_memcpy),
                 ("graph_launch", s_graph_launch), ("back_to_back", s_back_to_back), ("collectives", s_collectives), ("permute", s_permute)):
    if SECTIONS and name not in SECTIONS: continue
    section(name, fn)
# merge with a previous run's sections (partial re-runs via SECTIONS=...)
PREV = os.path.join(OUT, "hw_probe_measured.json")
if rank == 0 and SECTIONS and os.path.exists(PREV):
    old = json.load(open(PREV)).get("sections", {})
    for k, v in old.items():
        if k not in R["sections"] or "error" in R["sections"][k]: R["sections"][k] = v

# ----------------------------------------------------------------------------- derived table in the _HARDWARE_MEASUREMENTS schema
if rank == 0:
    S = R["sections"]; T = {}
    try:
        ops = S["xla_ops"]; empty = S["trivial"]["empty_mark_step"]["median_us"]
        VIEW = {"view", "reshape", "unsqueeze", "squeeze", "flatten", "narrow", "transpose", "permute", "expand", "contiguous", "slice", "split", "chunk"}
        T["xla_op_overhead_us_isolated"] = {k: round(max(v["isolated_1op_us"] - empty, 0.1), 1) for k, v in ops.items()}
        T["xla_op_overhead_us_marginal"] = {k: round(max(v["marginal_us"], 0.1), 2) for k, v in ops.items()}
        # table-style construction: isolated single-op graph cost for compute/copy ops, in-graph marginal for view-type ops
        T["xla_op_overhead_us"] = {k: (T["xla_op_overhead_us_marginal"][k] if k in VIEW else T["xla_op_overhead_us_isolated"][k]) for k in ops}
        m = S["memcpy"][max(S["memcpy"], key=lambda k: S["memcpy"][k]["bytes"])]
        T["memcpy_throughput"] = {"sequential_gbps": m["sequential_gbps"], "strided_gbps": m["strided_gbps_full_tensor_basis"],
                                  "clone_gbps": m["clone_gbps"], "strided_gbps_actual_bytes": m["strided_gbps_actual_bytes_basis"]}
        T["graph_launch_overhead_us"] = {"per_mark_step_us": S["graph_launch"]["per_mark_step_us_collective"],
                                         "per_mark_step_us_local_only": S["graph_launch"]["per_mark_step_us_local"]}
        b2b = S["back_to_back"]; T["back_to_back_amortization_us"] = {k: v for k, v in b2b.items() if not k.startswith("ag_")}
        d1 = b2b["dispatch_overhead_us_first_issue"]
        t1, t2, t8 = b2b["depth_1_total_us"], b2b["depth_2_total_us"], b2b["depth_8_total_us"]; t16 = b2b["depth_16_total_us"]
        alphas = (round(max(0.0, t2 - t1) / d1, 3), round(max(0.0, (t8 - t2) / 6.0) / d1, 3), round(max(0.0, (t16 - t8) / 8.0) / d1, 3))
        samples = []
        for nb in (65536, 524288, 4194304, 12582912, 50331648, 100663296):
            v = S["collectives"]["compile_first_exec_us"].get(str(nb))
            if isinstance(v, (int, float)): samples.append({"tensor_bytes": nb, "neff_load_us": round(v)})
        T["compilation_cost_us"] = {"samples": samples, "load_events_per_run": 2, "typical_training_steps": 5000}
        T["index_select_scaling"] = S["index_select"]
        T["all_gather_latency_us"] = S["collectives"]["all_gather"]; T["all_reduce_latency_us"] = S["collectives"]["all_reduce"]
        T["reduce_scatter_latency_us"] = S["collectives"]["reduce_scatter"]; T["collective_permute_latency_us"] = S["permute"]
        R["derived_table"] = T
        R["sim_params"] = dict(dispatch_overhead_us=d1, pipeline_amort_alpha1=alphas[0], pipeline_amort_alpha2=alphas[1], pipeline_amort_alpha3=alphas[2],
                               graph_launch_overhead_us=T["graph_launch_overhead_us"]["per_mark_step_us"],
                               memcpy_bytes_per_us=T["memcpy_throughput"]["strided_gbps"] * 1000.0, memcpy_seq_bytes_per_us=T["memcpy_throughput"]["sequential_gbps"] * 1000.0,
                               compilation_cost_samples=samples)
    except Exception as e:
        R["derived_error"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "hw_probe_measured.json"), "w"), indent=1); json.dump(R, open(os.path.join(OUT, f"hw_probe_measured_{time.strftime('%H%M%S')}.json"), "w"), indent=1)
    print(json.dumps(R.get("sim_params", R.get("derived_error")), indent=1), flush=True)
    print("Steady step: probe done", flush=True)
xm.rendezvous("probe_done")
