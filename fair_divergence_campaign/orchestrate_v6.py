"""v6 broad/diverse fair-gate sweep, PARALLELIZED.

Runs BOTH pipelines (SorcarCCL/kiss best-of-K seeds + OverlayCCL/strat once)
on every qualifying v6 problem under the IDENTICAL fair fp32 gate. Each
(problem, task) is an independent subprocess; a bounded thread pool launches
up to MAX_PAR at once (LLM-bound, so oversubscribe the 10 cores modestly).

Qualifying local set = the 9 v6 problems whose intended optimum clears 1.05x
sim headroom over the baseline (validate_v6b.py), spanning 4 distinct families:
FUSE, DEAD, ALG, SCALE. A2A/permute/deep-chain reframes cap below the local
single-collective floor (bandwidth-bound, not count-reducing) -> cluster track.

hd10_staged_shard_norm (verified 1.19x all 3 seeds) is folded into the final
tally from results_v2 without re-running (per user's criterion).
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
OUT = "/private/tmp/fair_diverge/results_v6"
GATE = os.environ.get("GATE_MODE", "fp32")
NODES = "7"
K_SEEDS = int(os.environ.get("K_SEEDS", "3"))
MAX_PAR = int(os.environ.get("MAX_PAR", "8"))

PROBLEMS = [
    "v6_fuse3", "v6_fuse6", "v6_fuse10",   # FUSE
    "v6_dead3", "v6_dead5",                # DEAD
    "v6_alg3", "v6_alg5",                  # ALG
    "v6_scale3", "v6_scale6",              # SCALE
]

_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def run(cmd, tag, logdir, env_extra=None):
    os.makedirs(logdir, exist_ok=True)
    lp = os.path.join(logdir, f"{tag}.log")
    with open(lp, "w") as lf:
        env = {**os.environ, "PYTHONPATH": ACC, "ACC_REPO": ACC}
        if env_extra:
            env.update(env_extra)
        t0 = time.time()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           env=env, text=True)
        dt = time.time() - t0
    return r.returncode, dt, lp


def do_overlay(prob, seed):
    # Overlay samples the LLM at temperature 0.8-1.0, so it varies run-to-run.
    # Run K_SEEDS independent draws (symmetric with kiss) and take the best.
    pd = os.path.join(OUT, prob)
    odir = os.path.join(pd, f"overlay_s{seed}")
    rc, dt, lp = run(
        [PY, f"{FD}/run_overlay_fair.py", "--problem", prob,
         "--pattern", "moe", "--num-nodes", NODES, "--gate", GATE,
         "--k", "5", "--rounds", "3", "--output-dir", odir],
        f"overlay_s{seed}", pd, env_extra={"OVERLAY_SEED": str(seed)})
    oj = os.path.join(odir, "overlay.json")
    if os.path.exists(oj):
        with open(oj) as f:
            od = json.load(f)
        return prob, f"overlay_s{seed}", {
            "seed": seed,
            "final_sim": od.get("final_sim"),
            "baseline_sim": od.get("baseline_sim"),
            "n_llm_calls": od.get("n_llm_calls"),
            "fell_back": od.get("fell_back_to_baseline", False),
            "wall_s": round(dt, 1)}
    return prob, f"overlay_s{seed}", {"seed": seed, "error": "no overlay.json",
                                      "log": lp, "rc": rc, "wall_s": round(dt, 1)}


def do_kiss(prob, seed):
    pd = os.path.join(OUT, prob)
    kdir = os.path.join(pd, f"kiss_s{seed}")
    rc, dt, lp = run(
        [PY, f"{FD}/run_kiss_fair.py", "--problem", prob,
         "--pattern", "moe", "--num-nodes", NODES, "--gate", GATE,
         "--max-budget", "1.5", "--max-steps", "30",
         "--output-dir", kdir],
        f"kiss_s{seed}", pd, env_extra={"KISS_SEED": str(seed)})
    kj = os.path.join(kdir, "kiss_summary.json")
    if os.path.exists(kj):
        with open(kj) as f:
            kd = json.load(f)
        return prob, f"kiss_s{seed}", {
            "seed": seed,
            "best_sim": kd.get("best_sim_time_us"),
            "baseline_sim": kd.get("baseline_sim_time_us"),
            "n_score_calls": kd.get("n_score_calls"),
            "n_ok": kd.get("n_ok"),
            "wall_s": round(dt, 1)}
    return prob, f"kiss_s{seed}", {"seed": seed, "error": "no summary",
                                   "log": lp, "rc": rc}


def main():
    os.makedirs(OUT, exist_ok=True)
    # build task list: overlay + K kiss seeds per problem
    tasks = []
    for prob in PROBLEMS:
        for s in range(K_SEEDS):
            tasks.append(("overlay", prob, s))
            tasks.append(("kiss", prob, s))

    results = {p: {"overlay_seeds": [], "kiss_seeds": []} for p in PROBLEMS}
    _log(f"launching {len(tasks)} tasks over {len(PROBLEMS)} problems, "
         f"MAX_PAR={MAX_PAR}, gate={GATE}, seeds={K_SEEDS}")

    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        futs = []
        for kind, prob, seed in tasks:
            if kind == "overlay":
                futs.append(ex.submit(do_overlay, prob, seed))
            else:
                futs.append(ex.submit(do_kiss, prob, seed))
        for fut in as_completed(futs):
            prob, tag, data = fut.result()
            if tag.startswith("overlay"):
                results[prob]["overlay_seeds"].append(data)
            else:
                results[prob]["kiss_seeds"].append(data)
            _log(f"  done {prob}/{tag}")

    # assemble report + verdicts
    report = {"gate": GATE, "nodes": NODES, "k_seeds": K_SEEDS,
              "problems": {}}
    for prob in PROBLEMS:
        entry = results[prob]
        # symmetric best-of-K: overlay's best draw vs kiss's best draw
        ovs = [o["final_sim"] for o in entry["overlay_seeds"]
               if isinstance(o.get("final_sim"), (int, float))]
        ov = min(ovs) if ovs else None
        ks = [k["best_sim"] for k in entry["kiss_seeds"]
              if isinstance(k.get("best_sim"), (int, float))]
        kiss_best = min(ks) if ks else None
        entry["kiss_best_sim"] = kiss_best
        entry["overlay_best_sim"] = ov
        entry["overlay_final_sim"] = ov  # back-compat key
        if ov and kiss_best:
            entry["sorcar_over_overlay"] = round(ov / kiss_best, 3)
            entry["diverges"] = (ov / kiss_best) >= 1.05
        report["problems"][prob] = entry
        _log(f"[{prob}] overlay={ov} kiss_best={kiss_best} "
             f"ratio={entry.get('sorcar_over_overlay')} "
             f"diverges={entry.get('diverges')}")

    # fold in hd10 (verified 1.19x, best-of-3, from results_v2)
    hd10 = _load_hd10()
    if hd10:
        report["problems"]["hd10_staged_shard_norm"] = hd10
        _log(f"[hd10_staged_shard_norm] (folded) "
             f"ratio={hd10.get('sorcar_over_overlay')} "
             f"diverges={hd10.get('diverges')}")

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _log("DONE")


def _load_hd10():
    src = "/private/tmp/fair_diverge/results_v2/report.json"
    if not os.path.exists(src):
        return None
    try:
        with open(src) as f:
            r2 = json.load(f)
        e = r2.get("problems", {}).get("hd10_staged_shard_norm")
        if e:
            e = dict(e)
            e["folded_from"] = src
        return e
    except Exception:
        return None


if __name__ == "__main__":
    main()
