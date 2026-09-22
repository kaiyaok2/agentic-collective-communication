"""v7 cost-aware divergence funnel.

Protocol (per user 2026-09-20):
  1. SCREEN every candidate at 1 seed each (overlay x1, kiss x1) under the fair
     fp32 gate. Compute ratio = overlay_sim / kiss_sim.
  2. DISCARD ties (ratio < 1.05) immediately -- no further spend.
  3. PROMOTE winners (ratio >= 1.05) to best-of-3 (overlay x3, kiss x3, take
     each side's best) to confirm the divergence is robust, not a 1-seed fluke.
  4. hd10_staged_shard_norm is a KNOWN winner -> always best-of-3 (symmetric).

This mirrors kiss's own best-of reporting while keeping overlay symmetric on
promoted problems, so a confirmed win can't be a best-of-1-vs-best-of-3
artifact. Screen phase is 2 runs/problem; only real candidates pay the 6-run
confirm.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PY = sys.executable
FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
OUT = "/private/tmp/fair_diverge/results_v7"
GATE = os.environ.get("GATE_MODE", "fp32")
NODES = "7"
MAX_PAR = int(os.environ.get("MAX_PAR", "8"))
PROMOTE = 1.05
CONFIRM_SEEDS = int(os.environ.get("CONFIRM_SEEDS", "3"))

CANDIDATES = [
    "v7_affine3", "v7_norm4", "v7_permsc", "v7_modsc",
    "v7_trisc", "v7_affine4", "v7_rsnorm3", "v7_rsnorm_mod",
]
ALWAYS_CONFIRM = ["hd10_staged_shard_norm"]

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


def overlay(prob, seed):
    pd = os.path.join(OUT, prob)
    odir = os.path.join(pd, f"overlay_s{seed}")
    rc, dt, lp = _run(
        [PY, f"{FD}/run_overlay_fair.py", "--problem", prob, "--pattern", "moe",
         "--num-nodes", NODES, "--gate", GATE, "--k", "5", "--rounds", "3",
         "--output-dir", odir], f"overlay_s{seed}", pd,
        env_extra={"OVERLAY_SEED": str(seed)})
    oj = os.path.join(odir, "overlay.json")
    if os.path.exists(oj):
        with open(oj) as f:
            od = json.load(f)
        return {"kind": "overlay", "prob": prob, "seed": seed,
                "sim": od.get("final_sim"), "baseline": od.get("baseline_sim"),
                "fell_back": od.get("fell_back_to_baseline", False),
                "n_llm": od.get("n_llm_calls"), "wall_s": round(dt, 1)}
    return {"kind": "overlay", "prob": prob, "seed": seed, "sim": None,
            "error": "no overlay.json", "log": lp, "rc": rc}


def kiss(prob, seed):
    pd = os.path.join(OUT, prob)
    kdir = os.path.join(pd, f"kiss_s{seed}")
    rc, dt, lp = _run(
        [PY, f"{FD}/run_kiss_fair.py", "--problem", prob, "--pattern", "moe",
         "--num-nodes", NODES, "--gate", GATE, "--max-budget", "1.5",
         "--max-steps", "30", "--output-dir", kdir], f"kiss_s{seed}", pd,
        env_extra={"KISS_SEED": str(seed)})
    kj = os.path.join(kdir, "kiss_summary.json")
    if os.path.exists(kj):
        with open(kj) as f:
            kd = json.load(f)
        return {"kind": "kiss", "prob": prob, "seed": seed,
                "sim": kd.get("best_sim_time_us"),
                "baseline": kd.get("baseline_sim_time_us"),
                "n_score": kd.get("n_score_calls"), "n_ok": kd.get("n_ok"),
                "wall_s": round(dt, 1)}
    return {"kind": "kiss", "prob": prob, "seed": seed, "sim": None,
            "error": "no summary", "log": lp, "rc": rc}


def _best(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return min(v) if v else None


def _submit(ex, jobs):
    futs = [ex.submit(fn, *a) for fn, a in jobs]
    out = []
    for fut in as_completed(futs):
        r = fut.result()
        out.append(r)
        _log(f"  done {r['prob']}/{r['kind']}_s{r['seed']} sim={r.get('sim')}")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"gate": GATE, "nodes": NODES, "promote_at": PROMOTE,
              "screen": {}, "confirmed": {}}

    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        # ---- Phase 1: 1-seed screen of all candidates ----
        _log(f"SCREEN: {len(CANDIDATES)} candidates x (overlay+kiss) @ 1 seed")
        screen_jobs = []
        for p in CANDIDATES:
            screen_jobs.append((overlay, (p, 0)))
            screen_jobs.append((kiss, (p, 0)))
        screen_res = _submit(ex, screen_jobs)

        by_prob = {}
        for r in screen_res:
            by_prob.setdefault(r["prob"], {})[r["kind"]] = r
        promoted = []
        for p in CANDIDATES:
            o = by_prob.get(p, {}).get("overlay", {})
            k = by_prob.get(p, {}).get("kiss", {})
            ov, ks = o.get("sim"), k.get("sim")
            ratio = round(ov / ks, 3) if (ov and ks) else None
            entry = {"overlay_s0": ov, "kiss_s0": ks, "ratio": ratio,
                     "overlay_fell_back": o.get("fell_back"),
                     "kiss_n_ok": k.get("n_ok")}
            report["screen"][p] = entry
            verdict = "PROMOTE" if (ratio and ratio >= PROMOTE) else "discard(tie)"
            if ratio and ratio >= PROMOTE:
                promoted.append(p)
            _log(f"[screen {p}] overlay={ov} kiss={ks} ratio={ratio} -> {verdict}")
        with open(os.path.join(OUT, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # ---- Phase 2: best-of-CONFIRM_SEEDS on winners + hd10 ----
        confirm_list = ALWAYS_CONFIRM + promoted
        _log(f"CONFIRM (best-of-{CONFIRM_SEEDS}): {confirm_list}")
        confirm_jobs = []
        for p in confirm_list:
            # reuse seed 0 for promoted candidates? re-run fresh for cleanliness
            for s in range(CONFIRM_SEEDS):
                confirm_jobs.append((overlay, (p, s)))
                confirm_jobs.append((kiss, (p, s)))
        confirm_res = _submit(ex, confirm_jobs) if confirm_jobs else []

        cby = {}
        for r in confirm_res:
            cby.setdefault(r["prob"], {"overlay": [], "kiss": []})[r["kind"]].append(r)
        for p in confirm_list:
            recs = cby.get(p, {"overlay": [], "kiss": []})
            ov = _best([x.get("sim") for x in recs["overlay"]])
            ks = _best([x.get("sim") for x in recs["kiss"]])
            ratio = round(ov / ks, 3) if (ov and ks) else None
            report["confirmed"][p] = {
                "overlay_best": ov, "kiss_best": ks, "ratio": ratio,
                "diverges": bool(ratio and ratio >= PROMOTE),
                "overlay_seeds": [x.get("sim") for x in recs["overlay"]],
                "kiss_seeds": [x.get("sim") for x in recs["kiss"]]}
            _log(f"[confirm {p}] overlay_best={ov} kiss_best={ks} ratio={ratio} "
                 f"diverges={report['confirmed'][p]['diverges']}")

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _log("DONE")


if __name__ == "__main__":
    main()
