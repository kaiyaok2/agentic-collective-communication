"""10-hour autonomous divergence campaign harness (reusable per round).

Invoked per round with a problem list. For each candidate:
  SCREEN (1 seed/side) -> ratio = overlay_sim / kiss_sim.
    discard if ratio < PROMOTE (tie) -- cheap.
  CONFIRM (N seeds/side) for promoted candidates -> full distributions +
    statistics so "significant" means significant:
      - best-of-N ratio (min overlay / min kiss)   [matches paper's best-of]
      - median ratio + bootstrap 95% CI
      - Mann-Whitney U p-value (kiss sims < overlay sims, one-sided)
    A candidate is a CONFIRMED DIVERGENCE only if best-of-N ratio >= PROMOTE
    AND p < 0.05 AND the CI lower bound > 1.0.

Appends every result to a persistent ledger so the 10h campaign accumulates
across rounds. Independent LLM temperature draws (no RNG seeding, no caching --
verified), so seeds are genuine iid samples.

Usage: campaign.py --round R<n> --problems p1,p2,... [--confirm-seeds 8]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

PY = sys.executable
FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
LEDGER = os.environ.get("LEDGER", "/private/tmp/fair_diverge/campaign_ledger.json")
GATE = os.environ.get("GATE_MODE", "fp32")
NODES = "7"
MAX_PAR = int(os.environ.get("MAX_PAR", "10"))
PROMOTE = 1.05

_lock = threading.Lock()


def _log(m):
    with _lock:
        print(m, flush=True)


def _run(cmd, tag, logdir, env_extra=None):
    os.makedirs(logdir, exist_ok=True)
    lp = os.path.join(logdir, f"{tag}.log")
    with open(lp, "w") as lf:
        env = {**os.environ, "PYTHONPATH": ACC, "ACC_REPO": ACC}
        if env_extra:
            env.update(env_extra)
        t0 = time.time()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, text=True)
        dt = time.time() - t0
    return r.returncode, dt, lp


def overlay(prob, seed, outroot):
    odir = os.path.join(outroot, prob, f"overlay_s{seed}")
    oj = os.path.join(odir, "overlay.json")
    if os.path.exists(oj):  # resume: reuse a valid cached seed result, skip the pipeline re-run
        try:
            with open(oj) as f:
                od = json.load(f)
            if od.get("final_sim") is not None:
                return {"kind": "overlay", "prob": prob, "seed": seed,
                        "sim": od.get("final_sim"), "baseline": od.get("baseline_sim"),
                        "fell_back": od.get("fell_back_to_baseline", False)}
        except (ValueError, OSError):
            pass  # corrupt/partial json -> fall through and re-run cleanly
    rc, dt, lp = _run(
        [PY, f"{FD}/run_overlay_fair.py", "--problem", prob, "--pattern", "moe",
         "--num-nodes", NODES, "--gate", GATE, "--k", "5", "--rounds", "3",
         "--output-dir", odir], f"overlay_s{seed}", os.path.join(outroot, prob),
        env_extra={"OVERLAY_SEED": str(seed)})
    oj = os.path.join(odir, "overlay.json")
    if os.path.exists(oj):
        with open(oj) as f:
            od = json.load(f)
        return {"kind": "overlay", "prob": prob, "seed": seed,
                "sim": od.get("final_sim"), "baseline": od.get("baseline_sim"),
                "fell_back": od.get("fell_back_to_baseline", False)}
    return {"kind": "overlay", "prob": prob, "seed": seed, "sim": None,
            "error": "no json", "log": lp, "rc": rc}


def kiss(prob, seed, outroot):
    kdir = os.path.join(outroot, prob, f"kiss_s{seed}")
    kj = os.path.join(kdir, "kiss_summary.json")
    if os.path.exists(kj):  # resume: reuse a valid cached seed result, skip the pipeline re-run
        try:
            with open(kj) as f:
                kd = json.load(f)
            if kd.get("best_sim_time_us") is not None:
                return {"kind": "kiss", "prob": prob, "seed": seed,
                        "sim": kd.get("best_sim_time_us"),
                        "baseline": kd.get("baseline_sim_time_us"),
                        "n_ok": kd.get("n_ok")}
        except (ValueError, OSError):
            pass  # corrupt/partial json -> fall through and re-run cleanly
    rc, dt, lp = _run(
        [PY, f"{FD}/run_kiss_fair.py", "--problem", prob, "--pattern", "moe",
         "--num-nodes", NODES, "--gate", GATE, "--max-budget", "1.5",
         "--max-steps", "30", "--output-dir", kdir], f"kiss_s{seed}",
        os.path.join(outroot, prob), env_extra={"KISS_SEED": str(seed)})
    kj = os.path.join(kdir, "kiss_summary.json")
    if os.path.exists(kj):
        with open(kj) as f:
            kd = json.load(f)
        return {"kind": "kiss", "prob": prob, "seed": seed,
                "sim": kd.get("best_sim_time_us"),
                "baseline": kd.get("baseline_sim_time_us"),
                "n_ok": kd.get("n_ok")}
    return {"kind": "kiss", "prob": prob, "seed": seed, "sim": None,
            "error": "no summary", "log": lp, "rc": rc}


def _mannwhitney_u(a, b):
    """One-sided Mann-Whitney U: P(a < b) large => a stochastically smaller.
    Returns (U, p_approx) using normal approximation with tie correction.
    Tests H1: kiss (a) sims are smaller than overlay (b) sims."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    # rank with ties averaged
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # tie-corrected sigma
    from collections import Counter
    cnt = Counter(v for v, _ in allv)
    N = n1 + n2
    tie = sum(t**3 - t for t in cnt.values())
    sigma = ((n1 * n2 / 12.0) * ((N + 1) - tie / (N * (N - 1)))) ** 0.5 if N > 1 else 0.0
    if sigma == 0:
        return u1, None
    # H1: a (kiss) smaller => a has LOW ranks => u1 small. z for u1 small.
    z = (u1 - mu) / sigma
    # one-sided p that kiss is smaller: P(U <= u1)
    import math
    p = 0.5 * math.erfc(-z / math.sqrt(2))  # P(Z <= z)
    return u1, round(p, 4)


def _bootstrap_ratio_ci(ov, ks, iters=2000):
    """Bootstrap 95% CI for median(overlay)/median(kiss). Deterministic LCG
    (no Math.random dependency); resample indices."""
    if not ov or not ks:
        return None, None
    seed = 12345
    def rnd(n):
        # Use HIGH bits: LCG low-order bits have a very short period (e.g.
        # `seed % 8` cycles with period 8), which would make every bootstrap
        # resample a fixed permutation of all indices -> zero-width CI. The
        # top bits mix properly.
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return (seed >> 16) % n
    ratios = []
    for _ in range(iters):
        bo = [ov[rnd(len(ov))] for _ in range(len(ov))]
        bk = [ks[rnd(len(ks))] for _ in range(len(ks))]
        mk = median(bk)
        if mk > 0:
            ratios.append(median(bo) / mk)
    if not ratios:
        return None, None
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios))]
    return round(lo, 4), round(hi, 4)


def _load_ledger():
    if os.path.exists(LEDGER):
        try:
            with open(LEDGER) as f:
                return json.load(f)
        except Exception:
            pass
    return {"rounds": {}, "confirmed_divergences": []}


def _save_ledger(led):
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--confirm-seeds", type=int, default=8)
    args = ap.parse_args()
    probs = [p.strip() for p in args.problems.split(",") if p.strip()]
    outroot = os.path.join(FD, f"results_{args.round}")
    os.makedirs(outroot, exist_ok=True)
    led = _load_ledger()
    rnd = led["rounds"].setdefault(args.round, {"screen": {}, "confirm": {}})

    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        # ---- SCREEN ----
        _log(f"[{args.round}] SCREEN {len(probs)} candidates @ 1 seed")
        jobs = []
        for p in probs:
            jobs.append(ex.submit(overlay, p, 0, outroot))
            jobs.append(ex.submit(kiss, p, 0, outroot))
        byp = {}
        for fut in as_completed(jobs):
            r = fut.result()
            byp.setdefault(r["prob"], {})[r["kind"]] = r
            _log(f"  screen done {r['prob']}/{r['kind']} sim={r.get('sim')}")
        promoted = []
        for p in probs:
            o = byp.get(p, {}).get("overlay", {}); k = byp.get(p, {}).get("kiss", {})
            ov, ks = o.get("sim"), k.get("sim")
            ratio = round(ov / ks, 3) if (ov and ks) else None
            rnd["screen"][p] = {"overlay_s0": ov, "kiss_s0": ks, "ratio": ratio,
                                "overlay_fell_back": o.get("fell_back")}
            if ratio and ratio >= PROMOTE:
                promoted.append(p)
                _log(f"[screen {p}] ratio={ratio} -> PROMOTE")
            else:
                _log(f"[screen {p}] ratio={ratio} -> discard")
        _save_ledger(led)

        # ---- CONFIRM (statistical) ----
        if promoted:
            N = args.confirm_seeds
            _log(f"[{args.round}] CONFIRM {promoted} @ {N} seeds/side")
            cjobs = []
            for p in promoted:
                for s in range(N):
                    cjobs.append(ex.submit(overlay, p, 100 + s, outroot))
                    cjobs.append(ex.submit(kiss, p, 100 + s, outroot))
            cby = {}
            for fut in as_completed(cjobs):
                r = fut.result()
                cby.setdefault(r["prob"], {"overlay": [], "kiss": []})[r["kind"]].append(r)
                _log(f"  confirm done {r['prob']}/{r['kind']}_s{r['seed']} sim={r.get('sim')}")
            for p in promoted:
                recs = cby.get(p, {"overlay": [], "kiss": []})
                ov = [x["sim"] for x in recs["overlay"] if isinstance(x.get("sim"), (int, float))]
                ks = [x["sim"] for x in recs["kiss"] if isinstance(x.get("sim"), (int, float))]
                bestratio = round(min(ov) / min(ks), 3) if (ov and ks) else None
                medratio = round(median(ov) / median(ks), 3) if (ov and ks) else None
                _, p_mw = _mannwhitney_u(ks, ov)  # H1: kiss smaller
                lo, hi = _bootstrap_ratio_ci(ov, ks)
                confirmed = bool(bestratio and bestratio >= PROMOTE and p_mw is not None
                                 and p_mw < 0.05 and lo is not None and lo > 1.0)
                rnd["confirm"][p] = {
                    "n": N, "overlay_sims": ov, "kiss_sims": ks,
                    "best_ratio": bestratio, "median_ratio": medratio,
                    "mannwhitney_p": p_mw, "ci95": [lo, hi],
                    "CONFIRMED_DIVERGENCE": confirmed}
                _log(f"[confirm {p}] best={bestratio} median={medratio} "
                     f"p={p_mw} ci=[{lo},{hi}] CONFIRMED={confirmed}")
                if confirmed and p not in led["confirmed_divergences"]:
                    led["confirmed_divergences"].append(p)
                _save_ledger(led)
        else:
            _log(f"[{args.round}] no promotions; all ties")

    _save_ledger(led)
    _log(f"[{args.round}] DONE")


if __name__ == "__main__":
    main()
