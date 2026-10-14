#!/usr/bin/env python3
"""Static _HARDWARE_MEASUREMENTS table (the snapshot the pipeline shipped with) vs the REAL on-device probe
(hw_probe_measured.json). Prints the side-by-side the rebuttal's A-Q3 answer needs.
Usage: python3 compare_table.py /tmp/ap7/pull/hwprobe/hw_probe_measured.json
"""
import ast, json, sys
src = open("/tmp/overlayccl_pub/search/agent_simulator_config.py").read()
i = src.index("_HARDWARE_MEASUREMENTS = {"); j = i + len("_HARDWARE_MEASUREMENTS = "); depth = 0
for k in range(j, len(src)):
    if src[k] == "{": depth += 1
    elif src[k] == "}":
        depth -= 1
        if depth == 0: end = k + 1; break
ST = ast.literal_eval(src[j:end])
P = json.load(open(sys.argv[1])); DT = P["derived_table"]; SP = P["sim_params"]; S = P["sections"]

def row(name, static, measured, note=""):
    print(f"{name:46s} {str(static):>14s} {str(measured):>14s}  {note}")
print(f"{'quantity':46s} {'static table':>14s} {'measured':>14s}")
print("-" * 90)
row("dispatch_overhead_us (AR 4KB, depth 1)", ST["back_to_back_amortization_us"]["dispatch_overhead_us_first_issue"], SP["dispatch_overhead_us"])
for d in (1, 2, 4, 8, 16, 32):
    k = f"depth_{d}_total_us"; row(f"back_to_back {k}", ST["back_to_back_amortization_us"].get(k), DT["back_to_back_amortization_us"].get(k))
row("pipeline alphas (a1,a2,a3)", "(.30,.10,.02)", f"({SP['pipeline_amort_alpha1']},{SP['pipeline_amort_alpha2']},{SP['pipeline_amort_alpha3']})")
row("graph_launch per_mark_step_us", ST["graph_launch_overhead_us"]["per_mark_step_us"], DT["graph_launch_overhead_us"]["per_mark_step_us"],
    f"(local-only graphs: {DT['graph_launch_overhead_us']['per_mark_step_us_local_only']})")
row("memcpy sequential GB/s (permute.contiguous)", ST["memcpy_throughput"]["sequential_gbps"], DT["memcpy_throughput"]["sequential_gbps"], f"clone {DT['memcpy_throughput']['clone_gbps']}")
row("memcpy strided GB/s (full-tensor basis)", ST["memcpy_throughput"]["strided_gbps"], DT["memcpy_throughput"]["strided_gbps"], f"actual-bytes basis {DT['memcpy_throughput']['strided_gbps_actual_bytes']}")
print("-- xla op overhead (us): static | measured isolated 1-op graph | measured in-graph marginal | table-style pick")
for op in ST["xla_op_overhead_us"]:
    if op in DT["xla_op_overhead_us"]:
        print(f"   {op:22s} {ST['xla_op_overhead_us'][op]:>8} {DT['xla_op_overhead_us_isolated'][op]:>10} {DT['xla_op_overhead_us_marginal'][op]:>10} {DT['xla_op_overhead_us'][op]:>10}")
for op in ("narrow_nonlead_contiguous", "permute_contiguous", "add"):
    if op in DT["xla_op_overhead_us"]:
        print(f"   {op:22s} {'-':>8} {DT['xla_op_overhead_us_isolated'][op]:>10} {DT['xla_op_overhead_us_marginal'][op]:>10} {DT['xla_op_overhead_us'][op]:>10}")
print("-- compilation cost samples (tensor_bytes -> first-exec us): static | measured")
sm = {s["tensor_bytes"]: s["neff_load_us"] for s in ST["compilation_cost_us"]["samples"]}; mm = {s["tensor_bytes"]: s["neff_load_us"] for s in SP["compilation_cost_samples"]}
for b in sorted(set(sm) | set(mm)): row(f"   {b}", sm.get(b), mm.get(b))
print("-- index_select scaling (us)")
for k in ST["index_select_scaling"]:
    if not isinstance(ST["index_select_scaling"][k], str): row(f"   {k}", ST["index_select_scaling"][k], DT["index_select_scaling"].get(k))
print("-- all_gather latency (us), world 32")
for k in ST["all_gather_latency_us"]: row(f"   {k}", ST["all_gather_latency_us"][k], DT["all_gather_latency_us"].get(k))
print("-- collective_permute latency (us)")
for k in ST["collective_permute_latency_us"]: row(f"   {k}", ST["collective_permute_latency_us"][k], DT["collective_permute_latency_us"].get(k))
