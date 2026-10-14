#!/usr/bin/env python3
"""Derive the _HARDWARE_MEASUREMENTS-schema table + simulator params from a (partial) probe JSON — same code as the
tail of train_hwprobe_7node.py (the on-device run was killed in the collective_permute section, which deadlocks
in CCOM bootstrap at ws=32 on this SDK)."""
import json, sys
P = json.load(open(sys.argv[1])); S = P["sections"]; T = {}
ops = S["xla_ops"]; empty = S["trivial"]["empty_mark_step"]["median_us"]
VIEW = {"view", "reshape", "unsqueeze", "squeeze", "flatten", "narrow", "transpose", "permute", "expand", "contiguous", "slice", "split", "chunk"}
T["xla_op_overhead_us_isolated"] = {k: round(max(v["isolated_1op_us"] - empty, 0.1), 1) for k, v in ops.items()}
T["xla_op_overhead_us_marginal"] = {k: round(max(v["marginal_us"], 0.1), 2) for k, v in ops.items()}
T["xla_op_overhead_us"] = {k: (T["xla_op_overhead_us_marginal"][k] if k in VIEW else T["xla_op_overhead_us_isolated"][k]) for k in ops}
m = S["memcpy"][max(S["memcpy"], key=lambda k: S["memcpy"][k]["bytes"])]
T["memcpy_throughput"] = {"sequential_gbps": m["sequential_gbps"], "strided_gbps": m["strided_gbps_full_tensor_basis"],
                          "clone_gbps": m["clone_gbps"], "strided_gbps_actual_bytes": m["strided_gbps_actual_bytes_basis"]}
T["graph_launch_overhead_us"] = {"per_mark_step_us": S["graph_launch"]["per_mark_step_us_collective"],
                                 "per_mark_step_us_local_only": S["graph_launch"]["per_mark_step_us_local"]}
b2b = S["back_to_back"]; T["back_to_back_amortization_us"] = {k: v for k, v in b2b.items() if not k.startswith("ag_")}
d1 = b2b["dispatch_overhead_us_first_issue"]
t1, t2, t8, t16 = b2b["depth_1_total_us"], b2b["depth_2_total_us"], b2b["depth_8_total_us"], b2b["depth_16_total_us"]
alphas = (round(max(0.0, t2 - t1) / d1, 3), round(max(0.0, (t8 - t2) / 6.0) / d1, 3), round(max(0.0, (t16 - t8) / 8.0) / d1, 3))
samples = []
for nb in (65536, 524288, 4194304, 12582912, 50331648, 100663296):
    v = S["collectives"]["compile_first_exec_us"].get(str(nb))
    if isinstance(v, (int, float)): samples.append({"tensor_bytes": nb, "neff_load_us": round(v)})
T["compilation_cost_us"] = {"samples": samples, "load_events_per_run": 2, "typical_training_steps": 5000}
T["index_select_scaling"] = S["index_select"]
T["all_gather_latency_us"] = S["collectives"]["all_gather"]; T["all_reduce_latency_us"] = S["collectives"]["all_reduce"]
T["reduce_scatter_latency_us"] = S["collectives"]["reduce_scatter"]
T["collective_permute_latency_us"] = {"note": "NOT MEASURED: xm.collective_permute deadlocks in CCOM bootstrap at ws=32 on SDK 2.22/2.29 (probe killed after 8 min)"}
P["derived_table"] = T
P["sim_params"] = dict(dispatch_overhead_us=d1, pipeline_amort_alpha1=alphas[0], pipeline_amort_alpha2=alphas[1], pipeline_amort_alpha3=alphas[2],
                       graph_launch_overhead_us=T["graph_launch_overhead_us"]["per_mark_step_us"],
                       memcpy_bytes_per_us=T["memcpy_throughput"]["strided_gbps"] * 1000.0, memcpy_seq_bytes_per_us=T["memcpy_throughput"]["sequential_gbps"] * 1000.0,
                       compilation_cost_samples=samples)
json.dump(P, open(sys.argv[2], "w"), indent=1); print(json.dumps(P["sim_params"]))
