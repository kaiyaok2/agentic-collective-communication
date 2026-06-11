#!/usr/bin/env python3
"""Build a baseline-vs-agent comparison from two OLMoE-10B run JSONs."""
import json, sys, os

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/agentic-collective-communication/training/results/olmoe_7node"
OUT = os.path.join(RESULTS_DIR, "COMPARISON.md")

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

base = load(os.path.join(RESULTS_DIR, "olmoe10b_baseline.json"))
agent = load(os.path.join(RESULTS_DIR, "olmoe10b_agent.json"))

if base is None or agent is None:
    print(f"MISSING: baseline={base is not None} agent={agent is not None}")
    sys.exit(1)

def stat(d):
    return dict(
        wall=d["wall_clock_s"],
        avg_ms=d["avg_step_ms"],
        steady_ms=d.get("steady_avg_step_ms") or d["avg_step_ms"],
        min_ms=d["min_step_ms"],
        max_ms=d["max_step_ms"],
        final_loss=d.get("final_loss"),
        finite=d.get("finite_loss_count"),
        total=len(d.get("step_times_ms", [])),
    )

b = stat(base)
a = stat(agent)

speedup_avg = b["avg_ms"] / a["avg_ms"] if a["avg_ms"] else float("nan")
speedup_steady = b["steady_ms"] / a["steady_ms"] if a["steady_ms"] else float("nan")
speedup_wall = b["wall"] / a["wall"] if a["wall"] else float("nan")

md = []
md.append("# OLMoE-10B 7-node baseline vs agent comparison\n")
md.append(f"World size: {base['world_size']} ranks (7x trn1.32xlarge)\n")
md.append(f"Architecture: {base['arch']}\n")
md.append(f"Steps: {base['steps']}  (baseline) / {agent['steps']} (agent)\n")
md.append("")
md.append("| Backend | Wall clock | Avg step | Steady-state step (>200) | Min/Max | Final loss | Finite/Total |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
md.append(f"| baseline | {b['wall']:.1f}s ({b['wall']/60:.1f} min) | {b['avg_ms']:.1f} ms | {b['steady_ms']:.1f} ms | {b['min_ms']:.0f} / {b['max_ms']:.0f} ms | {b['final_loss']} | {b['finite']}/{b['total']} |")
md.append(f"| **agent** | {a['wall']:.1f}s ({a['wall']/60:.1f} min) | {a['avg_ms']:.1f} ms | {a['steady_ms']:.1f} ms | {a['min_ms']:.0f} / {a['max_ms']:.0f} ms | {a['final_loss']} | {a['finite']}/{a['total']} |")
md.append("")
md.append(f"**Agent vs baseline speedups**")
md.append(f"- Wall-clock: **{speedup_wall:.3f}x**")
md.append(f"- Avg step:   **{speedup_avg:.3f}x**")
md.append(f"- Steady-state step (after warmup): **{speedup_steady:.3f}x**")
md.append("")
md.append("Backends call three collective primitives in the same model graph:")
md.append("- AllToAllV — MoE expert dispatch + combine")
md.append("- Multi-Tensor AllGather — N=8-slot expert weight prefetch (instrumented)")
md.append("- Fused ReduceScatter — gradient sync on the replicated params")
md.append("")
md.append("Baseline implementations: AG+RS for alltoallv, per-slot AG for multi-AG, per-tensor RS for fused-RS.")
md.append("Agent implementation: see runtime/trainium_alltoallv_7node.py (7-node search output) or runtime/trainium_alltoallv.py (1-node fallback).")

with open(OUT, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"wrote {OUT}")
print("\n".join(md))
