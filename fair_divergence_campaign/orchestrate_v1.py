"""Run BOTH pipelines (SorcarCCL/kiss + OverlayCCL/strat) on every hard
divergence problem under the IDENTICAL fair gate, collect sim results, and
write a divergence report. Kiss is stochastic -> run K_SEEDS seeds and take
the best (the paper reports best-of; the search is what we're measuring).

Everything is launched with THIS interpreter (sys.executable) so the scorer
subprocess inherits the same torch/anthropic env.
"""
import json
import os
import subprocess
import sys
import time

PY = sys.executable
FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
OUT = "/private/tmp/fair_diverge/results_v1"
GATE = os.environ.get("GATE_MODE", "fp32")
NODES = "7"
K_SEEDS = int(os.environ.get("K_SEEDS", "3"))

PROBLEMS = [
    "hd1_staged_dead_fold",
    "hd2_perblock_mixed_fold",
    "hd3_rs_ladder",
    "hd4_dead_slab_linear",
    "hd5_telescoping",
]


def run(cmd, tag, logdir):
    os.makedirs(logdir, exist_ok=True)
    lp = os.path.join(logdir, f"{tag}.log")
    with open(lp, "w") as lf:
        env = {**os.environ, "PYTHONPATH": ACC, "ACC_REPO": ACC}
        t0 = time.time()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           env=env, text=True)
        dt = time.time() - t0
    return r.returncode, dt, lp


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"gate": GATE, "nodes": NODES, "k_seeds": K_SEEDS, "problems": {}}
    for prob in PROBLEMS:
        pd = os.path.join(OUT, prob)
        os.makedirs(pd, exist_ok=True)
        entry = {"overlay": None, "kiss_seeds": [], "baseline_sim": None,
                 "opt_sim": None}

        # --- OverlayCCL (deterministic-ish; run once) ---
        odir = os.path.join(pd, "overlay")
        rc, dt, lp = run(
            [PY, f"{FD}/run_overlay_fair.py", "--problem", prob,
             "--pattern", "moe", "--num-nodes", NODES, "--gate", GATE,
             "--k", "5", "--rounds", "3", "--output-dir", odir],
            "overlay", pd)
        oj = os.path.join(odir, "overlay.json")
        if os.path.exists(oj):
            with open(oj) as f:
                od = json.load(f)
            entry["overlay"] = {
                "final_sim": od.get("final_sim"),
                "baseline_sim": od.get("baseline_sim"),
                "n_llm_calls": od.get("n_llm_calls"),
                "fell_back": od.get("fell_back_to_baseline", False),
                "wall_s": round(dt, 1)}
            entry["baseline_sim"] = od.get("baseline_sim")
        else:
            entry["overlay"] = {"error": "no overlay.json", "log": lp,
                                "rc": rc, "wall_s": round(dt, 1)}

        # --- SorcarCCL (kiss) K seeds ---
        for s in range(K_SEEDS):
            kdir = os.path.join(pd, f"kiss_s{s}")
            env_extra = {"KISS_SEED": str(s)}
            os.environ.update(env_extra)
            rc, dt, lp = run(
                [PY, f"{FD}/run_kiss_fair.py", "--problem", prob,
                 "--pattern", "moe", "--num-nodes", NODES, "--gate", GATE,
                 "--max-budget", "1.5", "--max-steps", "30",
                 "--output-dir", kdir],
                f"kiss_s{s}", pd)
            kj = os.path.join(kdir, "kiss_summary.json")
            if os.path.exists(kj):
                with open(kj) as f:
                    kd = json.load(f)
                entry["kiss_seeds"].append({
                    "seed": s,
                    "best_sim": kd.get("best_sim_time_us"),
                    "baseline_sim": kd.get("baseline_sim_time_us"),
                    "n_score_calls": kd.get("n_score_calls"),
                    "n_ok": kd.get("n_ok"),
                    "wall_s": round(dt, 1)})
            else:
                entry["kiss_seeds"].append({"seed": s, "error": "no summary",
                                            "log": lp, "rc": rc})

        # --- divergence verdict ---
        ov = entry["overlay"].get("final_sim") if entry["overlay"] else None
        ks = [k["best_sim"] for k in entry["kiss_seeds"]
              if isinstance(k.get("best_sim"), (int, float))]
        kiss_best = min(ks) if ks else None
        entry["kiss_best_sim"] = kiss_best
        entry["overlay_final_sim"] = ov
        if ov and kiss_best:
            entry["sorcar_over_overlay"] = round(ov / kiss_best, 3)
            entry["diverges"] = (ov / kiss_best) >= 1.05
        report["problems"][prob] = entry
        with open(os.path.join(OUT, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[{prob}] overlay={ov} kiss_best={kiss_best} "
              f"ratio={entry.get('sorcar_over_overlay')} "
              f"diverges={entry.get('diverges')}", flush=True)

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
