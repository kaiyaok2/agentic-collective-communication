"""Confirm-only driver: SKIP the fragile 1-seed screen (where overlay's single draw
occasionally gate-fails -> sim=None -> spurious discard) and run the statistical
CONFIRM directly on the named problems. Reuses campaign.py's overlay/kiss workers,
Mann-Whitney, bootstrap CI, and the SAME confirmation criterion & ledger.

Usage: confirm_only.py --round Rn --problems p1,p2,... [--confirm-seeds 16]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import campaign as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--confirm-seeds", type=int, default=16)
    args = ap.parse_args()
    probs = [p.strip() for p in args.problems.split(",") if p.strip()]
    outroot = os.path.join(C.FD, f"results_{args.round}")
    os.makedirs(outroot, exist_ok=True)
    led = C._load_ledger()
    rnd = led["rounds"].setdefault(args.round, {"screen": {}, "confirm": {}})
    N = args.confirm_seeds

    with ThreadPoolExecutor(max_workers=C.MAX_PAR) as ex:
        C._log(f"[{args.round}] CONFIRM-ONLY (no screen) {probs} @ {N} seeds/side")
        cjobs = []
        for p in probs:
            for s in range(N):
                cjobs.append(ex.submit(C.overlay, p, 100 + s, outroot))
                cjobs.append(ex.submit(C.kiss, p, 100 + s, outroot))
        cby = {}
        for fut in as_completed(cjobs):
            r = fut.result()
            cby.setdefault(r["prob"], {"overlay": [], "kiss": []})[r["kind"]].append(r)
            C._log(f"  confirm done {r['prob']}/{r['kind']}_s{r['seed']} sim={r.get('sim')}")
        for p in probs:
            recs = cby.get(p, {"overlay": [], "kiss": []})
            ov = [x["sim"] for x in recs["overlay"] if isinstance(x.get("sim"), (int, float))]
            ks = [x["sim"] for x in recs["kiss"] if isinstance(x.get("sim"), (int, float))]
            bestratio = round(min(ov) / min(ks), 3) if (ov and ks) else None
            medratio = round(median(ov) / median(ks), 3) if (ov and ks) else None
            _, p_mw = C._mannwhitney_u(ks, ov)
            lo, hi = C._bootstrap_ratio_ci(ov, ks)
            confirmed = bool(bestratio and bestratio >= C.PROMOTE and p_mw is not None
                             and p_mw < 0.05 and lo is not None and lo > 1.0)
            rnd["confirm"][p] = {
                "n": N, "n_overlay_ok": len(ov), "n_kiss_ok": len(ks),
                "overlay_sims": ov, "kiss_sims": ks,
                "best_ratio": bestratio, "median_ratio": medratio,
                "mannwhitney_p": p_mw, "ci95": [lo, hi],
                "CONFIRMED_DIVERGENCE": confirmed}
            C._log(f"[confirm {p}] n_ov={len(ov)} n_ks={len(ks)} best={bestratio} "
                   f"median={medratio} p={p_mw} ci=[{lo},{hi}] CONFIRMED={confirmed}")
            if confirmed and p not in led["confirmed_divergences"]:
                led["confirmed_divergences"].append(p)
            C._save_ledger(led)
    C._save_ledger(led)
    C._log(f"[{args.round}] DONE")


if __name__ == "__main__":
    main()
