"""Append a CONFIRMED-DIVERGENCE problem's full 9-seed artifacts into
CONFIRMED_WINS_ARTIFACTS/family<N>_9seed/<name>/, replicating the verified
r60_b5_L2_d8 layout exactly:

  <name>/
    overlay_s100..108/   (full run dir: best_code.py + overlay.json + any trajectory files)
    overlay_s100..108.log
    kiss_s100..108/       (full run dir: best_code.py + kiss_summary.json + any trajectory files)
    kiss_s100..108.log
    MANIFEST.json         (from the round ledger's confirm entry)

Refuses to append unless the ledger marks the problem CONFIRMED_DIVERGENCE.
Copies FULL seed dirs (trajectories) + logs -- nothing dropped.

Usage:
  append_confirmed.py --family 1 --round r59bfam1 \
      --ledger /private/tmp/fair_diverge/ledger_fam1.json \
      --results /private/tmp/fair_diverge/results_r59bfam1 \
      --problems r59b_su_a5_d8_p1024,... [--name-map src=dst,...] [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "CONFIRMED_WINS_ARTIFACTS")
SEEDS = list(range(100, 109))  # 9 confirm seeds


def _load(path):
    with open(path) as f:
        return json.load(f)


def append_one(family, round_name, ledger, results_dir, prob, dst_name, dry):
    fam_dir = os.path.join(ART, f"family{family}_9seed")
    conf = ledger.get("rounds", {}).get(round_name, {}).get("confirm", {}).get(prob)
    if conf is None:
        return f"SKIP {prob}: no confirm entry in round {round_name}"
    if not conf.get("CONFIRMED_DIVERGENCE"):
        return (f"SKIP {prob}: NOT confirmed "
                f"(best={conf.get('best_ratio')} p={conf.get('mannwhitney_p')} "
                f"ci={conf.get('ci95')})")
    src = os.path.join(results_dir, prob)
    if not os.path.isdir(src):
        return f"SKIP {prob}: source dir missing {src}"
    dst = os.path.join(fam_dir, dst_name)
    if os.path.exists(dst):
        return f"SKIP {prob}: destination already exists {dst}"

    # verify all 18 seed dirs + logs present before touching anything
    missing = []
    for side in ("overlay", "kiss"):
        for s in SEEDS:
            d = os.path.join(src, f"{side}_s{s}")
            lg = os.path.join(src, f"{side}_s{s}.log")
            if not os.path.isdir(d):
                missing.append(d)
            if not os.path.exists(lg):
                missing.append(lg)
    if missing:
        return f"SKIP {prob}: incomplete artifacts, missing {len(missing)} paths e.g. {missing[0]}"

    if dry:
        return (f"DRY  {prob} -> {dst_name}: would append 18 seed dirs + 18 logs "
                f"(best={conf['best_ratio']} median={conf['median_ratio']} "
                f"p={conf['mannwhitney_p']} ci={conf['ci95']})")

    os.makedirs(dst, exist_ok=False)
    for side in ("overlay", "kiss"):
        for s in SEEDS:
            shutil.copytree(os.path.join(src, f"{side}_s{s}"),
                            os.path.join(dst, f"{side}_s{s}"))
            shutil.copy2(os.path.join(src, f"{side}_s{s}.log"),
                         os.path.join(dst, f"{side}_s{s}.log"))
    manifest = {
        "problem": dst_name, "round": round_name, "n": conf.get("n", 9),
        "best_ratio": conf.get("best_ratio"),
        "median_ratio": conf.get("median_ratio"),
        "mannwhitney_p": conf.get("mannwhitney_p"),
        "ci95": conf.get("ci95"),
        "overlay_sims": conf.get("overlay_sims"),
        "kiss_sims": conf.get("kiss_sims"),
        "confirmed": True,
    }
    with open(os.path.join(dst, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return (f"APPENDED {prob} -> family{family}_9seed/{dst_name} "
            f"(best={conf['best_ratio']} median={conf['median_ratio']} "
            f"p={conf['mannwhitney_p']} ci={conf['ci95']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--name-map", default="",
                    help="comma list src=dst to rename appended dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ledger = _load(args.ledger)
    nm = {}
    for pair in args.name_map.split(","):
        if "=" in pair:
            s, d = pair.split("=", 1)
            nm[s.strip()] = d.strip()
    probs = [p.strip() for p in args.problems.split(",") if p.strip()]
    for prob in probs:
        dst_name = nm.get(prob, prob)
        print(append_one(args.family, args.round, ledger, args.results,
                         prob, dst_name, args.dry_run))


if __name__ == "__main__":
    main()
