"""Merge warm-cache RT jsons into a 3-column (baseline/strat/sorcar) table.

Reads /home/ubuntu/rt_sweep/<problem>.<tag>.json (tag in baseline|strat|sorcar),
computes sorcar-vs-strat and sorcar-vs-baseline warm speedups, and drops any
problem with NO warm-RT divergence (sorcar within DIV_THRESH of strat).
"""
import os
import json
import glob

RT_DIR = "/home/ubuntu/rt_sweep"
NAMES = "/home/ubuntu/divergent_56.txt"
DIV_THRESH = 0.05  # sorcar must beat strat by >5% warm to count as divergent

names = [l.strip() for l in open(NAMES) if l.strip()]
rows = {}
for nm in names:
    rec = {}
    for tag in ("baseline", "strat", "sorcar"):
        f = os.path.join(RT_DIR, "%s.%s.json" % (nm, tag))
        if os.path.exists(f):
            try:
                rec[tag] = json.load(open(f))
            except Exception:
                rec[tag] = None
    rows[nm] = rec


def warm(rec, tag):
    r = rec.get(tag)
    if r and isinstance(r, dict) and "warm_ms" in r:
        return r["warm_ms"]
    return None


# Sorcar candidates that pass the sim gate but ABORT on real 224-rank HW
# (reduce_scatter shard_count=224 on tensors not divisible by 224). Deterministic
# (reproduced 2x). Classified as sorcar RT-fails, not wins and not strat wins.
RT_FAIL = {"sequential_ar_chain_edge_chal", "three_group_dead_verify_chal"}

complete, incomplete, rt_fail = [], [], []
for nm in names:
    rec = rows[nm]
    b, s, c = warm(rec, "baseline"), warm(rec, "strat"), warm(rec, "sorcar")
    if nm in RT_FAIL and warm(rec, "sorcar") is None:
        rt_fail.append(nm)
        continue
    if None in (b, s, c):
        incomplete.append(nm)
        continue
    complete.append((nm, b, s, c))

# analysis
div_wins, no_div = [], []
for nm, b, s, c in complete:
    sp_vs_strat = s / c if c > 0 else float("inf")
    sp_vs_base = b / c if c > 0 else float("inf")
    entry = {"problem": nm, "baseline_ms": round(b, 4), "strat_ms": round(s, 4),
             "sorcar_ms": round(c, 4), "sorcar_vs_strat": round(sp_vs_strat, 3),
             "sorcar_vs_baseline": round(sp_vs_base, 3)}
    if sp_vs_strat > 1.0 + DIV_THRESH:
        div_wins.append(entry)
    else:
        no_div.append(entry)

div_wins.sort(key=lambda e: -e["sorcar_vs_strat"])
out = {"complete": len(complete), "incomplete": incomplete,
       "rt_fail_sorcar": rt_fail,
       "n_divergent": len(div_wins), "n_no_divergence": len(no_div),
       "divergent": div_wins, "no_divergence": no_div}
json.dump(out, open("/home/ubuntu/rt_three_col.json", "w"), indent=2)

print("complete=%d incomplete=%d rt_fail_sorcar=%d" %
      (len(complete), len(incomplete), len(rt_fail)))
if rt_fail:
    print("  sorcar RT-fail (sim-pass, HW-abort — reduce_scatter/224):",
          " ".join(rt_fail))
if incomplete:
    print("  still-missing:", " ".join(incomplete[:12]),
          "..." if len(incomplete) > 12 else "")
print("warm-RT divergent (sorcar>strat by >%.0f%%): %d" % (DIV_THRESH * 100, len(div_wins)))
print("no-divergence (dropped): %d" % len(no_div))
print()
print("%-38s %9s %9s %9s %8s %8s" %
      ("problem", "base_ms", "strat_ms", "sorc_ms", "vsStrat", "vsBase"))
for e in div_wins:
    print("%-38s %9.3f %9.3f %9.3f %7.2fx %7.2fx" %
          (e["problem"], e["baseline_ms"], e["strat_ms"], e["sorcar_ms"],
           e["sorcar_vs_strat"], e["sorcar_vs_baseline"]))
if no_div:
    print("\n-- no warm-RT divergence (dropped) --")
    for e in no_div:
        print("%-38s %9.3f %9.3f %9.3f %7.2fx" %
              (e["problem"], e["baseline_ms"], e["strat_ms"], e["sorcar_ms"],
               e["sorcar_vs_strat"]))
