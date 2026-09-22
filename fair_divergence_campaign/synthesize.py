"""Final synthesis: aggregate all campaign rounds into one verdict.

For each round log present, recompute fixed-bootstrap stats, then classify each
problem as CONFIRMED (Sorcar>Overlay: best-of-N>=1.05 & MW-p<0.05 & CI_lo>1.0),
DISTRIBUTIONAL (median>1 & p<.05 but best-of-N ties), or TIE. Also emit the
collective-count mechanism table and the depth->divergence curve.
"""
import re, sys, glob, math, os
from statistics import median
from collections import defaultdict

FD = "/private/tmp/fair_diverge"
PROMOTE = 1.05


def rnd_gen(seed):
    s = [seed]
    def r(n):
        s[0] = (1103515245 * s[0] + 12345) & 0x7FFFFFFF
        return (s[0] >> 16) % n
    return r


def bootstrap(ov, ks, iters=2000):
    r = rnd_gen(12345); ratios = []
    for _ in range(iters):
        bo = [ov[r(len(ov))] for _ in range(len(ov))]
        bk = [ks[r(len(ks))] for _ in range(len(ks))]
        mk = median(bk)
        if mk > 0:
            ratios.append(median(bo) / mk)
    ratios.sort()
    return round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios))], 4)


def mannwhitney(a, b):
    na, nb = len(a), len(b)
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * (na + nb); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    Ra = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 0)
    Ua = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    tie = 0; i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        t = j - i + 1; tie += t ** 3 - t; i = j + 1
    N = na + nb
    var = na * nb / 12.0 * ((N + 1) - tie / (N * (N - 1)))
    if var <= 0:
        return 1.0
    z = (Ua - mu + 0.5) / math.sqrt(var)
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))), 4)


def parse(logpath):
    log = open(logpath).read()
    pat = re.compile(r'confirm done (\S+?)/(overlay|kiss)_s\d+ sim=([\d.]+)')
    data = defaultdict(lambda: {"overlay": [], "kiss": []})
    for prob, side, val in pat.findall(log):
        data[prob][side].append(float(val))
    return data


def classify(ov, ks):
    best = min(ov) / min(ks); med = median(ov) / median(ks)
    p = mannwhitney(ks, ov); lo, hi = bootstrap(ov, ks)
    if best >= PROMOTE and p < 0.05 and lo > 1.0:
        return "CONFIRMED", best, med, p, lo, hi
    if med > 1.0 and p < 0.05:
        return "distributional", best, med, p, lo, hi
    return "tie", best, med, p, lo, hi


def min_collectives(round_dir, prob, side):
    import subprocess
    best = 99
    for d in glob.glob(f"{round_dir}/{prob}/{side}_s1*"):
        f = os.path.join(d, "best_code.py")
        if not os.path.exists(f):
            continue
        code = open(f).read()
        c = len(re.findall(r'xm\.(all_reduce|all_gather|reduce_scatter|all_to_all|collective_permute)\(', code))
        best = min(best, c)
    return best if best < 99 else None


def main():
    rounds = sorted(glob.glob(f"{FD}/campaign_r*.log"),
                    key=lambda p: int(re.search(r'campaign_r(\d+)', p).group(1)))
    confirmed = []; distributional = []; ties = []
    print("=" * 92)
    print("CAMPAIGN SYNTHESIS — Sorcar (kiss) vs Overlay (strat), fair fp32 gate, symmetric best-of-8")
    print("=" * 92)
    for lp in rounds:
        rnd = re.search(r'(campaign_)(r\d+)', lp).group(2)
        data = parse(lp)
        if not data:
            continue
        rdir = f"{FD}/results_{rnd}"
        print(f"\n### {rnd}  ({os.path.basename(lp)})")
        print(f"{'problem':22}{'best':>7}{'med':>7}{'p':>8}{'ci_lo':>7}{'ci_hi':>7}  {'minC(ov/ks)':>12}  class")
        for prob in sorted(data):
            ov = data[prob]["overlay"]; ks = data[prob]["kiss"]
            if not ov or not ks:
                continue
            cls, best, med, p, lo, hi = classify(ov, ks)
            mo = min_collectives(rdir, prob, "overlay")
            mk = min_collectives(rdir, prob, "kiss")
            mstr = f"{mo}/{mk}"
            print(f"{prob:22}{best:>7.3f}{med:>7.3f}{p:>8.4f}{lo:>7.3f}{hi:>7.3f}  {mstr:>12}  {cls}")
            rec = (rnd, prob, best, med, p, lo, hi, mo, mk)
            if cls == "CONFIRMED":
                confirmed.append(rec)
            elif cls == "distributional":
                distributional.append(rec)
            else:
                ties.append(rec)
    print("\n" + "=" * 92)
    print(f"TOTALS: {len(confirmed)} CONFIRMED Sorcar>Overlay | "
          f"{len(distributional)} distributional-only | {len(ties)} tie | "
          f"0 Overlay>Sorcar confirmed")
    print("\nCONFIRMED divergences (best-of-8 >=1.05 & MW-p<.05 & bootstrap-CI_lo>1.0):")
    for rnd, prob, best, med, p, lo, hi, mo, mk in sorted(confirmed, key=lambda r: -r[2]):
        print(f"  {rnd:4} {prob:22} best={best:.3f} med={med:.3f} p={p:.4f} "
              f"CI[{lo:.3f},{hi:.3f}] minCollectives ov={mo} ks={mk}")
    # depth curve from deep* problems
    print("\nDEPTH -> divergence (deep_chain family, best-of-8 ratio):")
    depthmap = {}
    for rnd, prob, best, med, p, lo, hi, mo, mk in confirmed + distributional + ties:
        m = re.search(r'deep(\d+)(_big)?$', prob)
        if m:
            depthmap.setdefault(int(m.group(1)), []).append((prob, best, med))
    for d in sorted(depthmap):
        for prob, best, med in depthmap[d]:
            print(f"  depth={d:2}  {prob:18} best={best:.3f} med={med:.3f}")


if __name__ == "__main__":
    main()
