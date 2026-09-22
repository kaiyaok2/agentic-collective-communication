"""Recompute correct bootstrap CIs + best-of-N + Mann-Whitney from a campaign
round log (the in-memory bootstrap had an LCG-low-bit bug -> zero-width CIs;
Mann-Whitney p is unaffected). Usage: python recompute_ci.py campaign_r2.log
"""
import re, sys, math
from statistics import median
from collections import defaultdict

PROMOTE = 1.05


def rnd_gen(seed):
    s = [seed]
    def r(n):
        s[0] = (1103515245 * s[0] + 12345) & 0x7FFFFFFF
        return (s[0] >> 16) % n     # HIGH bits (low bits have period n)
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


def mannwhitney_u(a, b):
    # H1: a (kiss) stochastically SMALLER than b (overlay). tie-corrected normal.
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
    # tie correction
    tie = 0; i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        t = j - i + 1; tie += t ** 3 - t; i = j + 1
    N = na + nb
    sigma = math.sqrt(na * nb / 12.0 * ((N + 1) - tie / (N * (N - 1))))
    if sigma == 0:
        return 1.0
    # H1 kiss smaller => Ua small; one-sided
    z = (Ua - mu + 0.5) / sigma
    from math import erf
    p = 0.5 * (1 + erf(z / math.sqrt(2)))
    return round(p, 4)


def main(logpath):
    log = open(logpath).read()
    pat = re.compile(r'confirm done (\S+?)/(overlay|kiss)_s\d+ sim=([\d.]+)')
    data = defaultdict(lambda: {"overlay": [], "kiss": []})
    for prob, side, val in pat.findall(log):
        data[prob][side].append(float(val))
    if not data:
        print("no confirm data in", logpath); return
    print(f"{'problem':24}{'nO':>4}{'nK':>4}{'best':>7}{'med':>7}{'p_mw':>8}{'ci_lo':>8}{'ci_hi':>8}  VERDICT")
    for prob in sorted(data):
        ov = data[prob]["overlay"]; ks = data[prob]["kiss"]
        if not ov or not ks:
            continue
        best = min(ov) / min(ks); med = median(ov) / median(ks)
        p = mannwhitney_u(ks, ov)
        lo, hi = bootstrap(ov, ks)
        conf = (best >= PROMOTE) and (p < 0.05) and (lo > 1.0)
        print(f"{prob:24}{len(ov):>4}{len(ks):>4}{best:>7.3f}{med:>7.3f}{p:>8.4f}"
              f"{lo:>8.3f}{hi:>8.3f}  {'CONFIRMED' if conf else '-'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "campaign_r1.log")
