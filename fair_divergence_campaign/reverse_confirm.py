"""Symmetric BIDIRECTIONAL confirm for a candidate that screened as a possible
REVERSE (Overlay>Sorcar) signal. Runs N seeds/side and reports BOTH directions:
  forward ratio  = median(overlay)/median(kiss)   (Sorcar faster if >1)
  reverse ratio  = median(kiss)/median(overlay)    (Overlay faster if >1)
plus best-of-N in both directions, Mann-Whitney both tails, bootstrap CI.

A CONFIRMED reverse divergence requires best-of-N (min kiss / min overlay) >= 1.05
AND MW p(overlay<kiss) < 0.05 AND the reverse bootstrap CI lower > 1.0. Symmetric
to the forward criterion, so neither direction gets a free pass.

Usage: reverse_confirm.py <prob1,prob2,...> [N]
"""
import sys, os, json
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/private/tmp/fair_diverge")
import campaign as C  # reuse overlay(), kiss(), _mannwhitney_u, _bootstrap_ratio_ci

probs = [p.strip() for p in sys.argv[1].split(",") if p.strip()]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
outroot = os.path.join(C.FD, "results_reverse")
os.makedirs(outroot, exist_ok=True)


def _boot_ci(a, b, iters=2000):
    """Bootstrap 95% CI for median(a)/median(b). Reuses campaign's high-bit LCG."""
    if not a or not b:
        return None, None
    seed = 987654321
    def rnd(n):
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return (seed >> 16) % n
    r = []
    for _ in range(iters):
        ba = [a[rnd(len(a))] for _ in range(len(a))]
        bb = [b[rnd(len(b))] for _ in range(len(b))]
        mb = median(bb)
        if mb > 0:
            r.append(median(ba) / mb)
    if not r:
        return None, None
    r.sort()
    return round(r[int(0.025 * len(r))], 4), round(r[int(0.975 * len(r))], 4)


print(f"BIDIRECTIONAL confirm @ {N} seeds/side  (gate={C.GATE})")
for p in probs:
    with ThreadPoolExecutor(max_workers=C.MAX_PAR) as ex:
        jobs = []
        for s in range(N):
            jobs.append(ex.submit(C.overlay, p, 200 + s, outroot))
            jobs.append(ex.submit(C.kiss, p, 200 + s, outroot))
        ov, ks = [], []
        for fut in as_completed(jobs):
            r = fut.result()
            v = r.get("sim")
            if isinstance(v, (int, float)):
                (ov if r["kind"] == "overlay" else ks).append(v)
    if not ov or not ks:
        print(f"{p}: insufficient data ov={len(ov)} ks={len(ks)}")
        continue
    fwd_best = round(min(ov) / min(ks), 3)          # Sorcar faster if >1
    rev_best = round(min(ks) / min(ov), 3)          # Overlay faster if >1
    fwd_med = round(median(ov) / median(ks), 3)
    rev_med = round(median(ks) / median(ov), 3)
    # MW H1: kiss smaller (forward). p_fwd small => Sorcar faster.
    _, p_fwd = C._mannwhitney_u(ks, ov)
    # MW H1: overlay smaller (reverse). p_rev small => Overlay faster.
    _, p_rev = C._mannwhitney_u(ov, ks)
    rev_lo, rev_hi = _boot_ci(ks, ov)               # CI for median(kiss)/median(overlay)
    fwd_lo, fwd_hi = _boot_ci(ov, ks)
    rev_conf = bool(rev_best >= 1.05 and p_rev is not None and p_rev < 0.05
                    and rev_lo is not None and rev_lo > 1.0)
    fwd_conf = bool(fwd_best >= 1.05 and p_fwd is not None and p_fwd < 0.05
                    and fwd_lo is not None and fwd_lo > 1.0)
    print(f"\n=== {p} (nO={len(ov)} nK={len(ks)}) ===")
    print(f"  overlay sims: min={min(ov):.1f} med={median(ov):.1f} max={max(ov):.1f}")
    print(f"  kiss    sims: min={min(ks):.1f} med={median(ks):.1f} max={max(ks):.1f}")
    print(f"  FORWARD (Sorcar>Overlay): best={fwd_best} med={fwd_med} p={p_fwd} "
          f"CI[{fwd_lo},{fwd_hi}] CONFIRMED={fwd_conf}")
    print(f"  REVERSE (Overlay>Sorcar): best={rev_best} med={rev_med} p={p_rev} "
          f"CI[{rev_lo},{rev_hi}] CONFIRMED={rev_conf}")
