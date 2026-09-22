"""SorcarCCL (kiss ReAct) driver against the FAIR symmetric scorer.

Identical to experiments/ablation_kiss_vs_cc/sorcar_phase3_7node.py except
SCORE_PY points at score_service_fair.py and GATE_MODE is passed through.
"""
import argparse
import atexit
import json
import os
import pathlib
import subprocess
import sys
import time

ACC = os.environ.get("ACC_REPO", "/private/tmp/acc_verify")
KISS_SRC = os.environ.get("KISS_SRC", "/private/tmp/kiss_ai/src")
sys.path.insert(0, ACC)
sys.path.insert(0, KISS_SRC)
sys.path.insert(0, os.path.join(ACC, "experiments", "ablation_kiss_vs_cc"))

SCORE_PY = "/private/tmp/fair_diverge/score_service_fair.py"
PY = sys.executable

import kiss_token_shim  # noqa
from kiss.core.kiss_agent import KISSAgent

GENERIC_EVO_PROMPT = pathlib.Path(
    os.path.join(ACC, "prompts", "generic_evolution.md")).read_text()


def start_scorer(problem, pattern, num_nodes, gate_mode):
    env = os.environ.copy()
    env.update(SCORE_PROBLEM=problem, SCORE_PATTERN=pattern,
               SCORE_NUM_NODES=str(num_nodes), GATE_MODE=gate_mode,
               ACC_REPO=ACC, PYTHONPATH=ACC)
    p = subprocess.Popen([PY, SCORE_PY], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env, text=True, bufsize=1)
    while True:
        line = p.stderr.readline()
        if not line:
            raise RuntimeError("scorer died: " + (p.stderr.read() or ""))
        if "[score_service] ready" in line:
            break
    return p


def score_via_pipe(p, code):
    p.stdin.write(json.dumps({"code": code}) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    return json.loads(line) if line else {"ok": False, "error": "scorer EOF"}


def get_baseline_code(problem_name):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem_name}');"
         "tmpls = p.builtin_templates; k = next(iter(tmpls.keys()));"
         "print(json.dumps({'name': k, 'code': tmpls[k]}))"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": ACC})
    if r.returncode != 0:
        raise RuntimeError(f"baseline fetch failed: {r.stderr[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--pattern", default="moe")
    ap.add_argument("--num-nodes", type=int, default=7)
    ap.add_argument("--gate", default="fp32", choices=["fp32", "fp32_bf16"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-budget", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = get_baseline_code(args.problem)
    p = start_scorer(args.problem, args.pattern, args.num_nodes, args.gate)
    atexit.register(lambda: p.terminate() if p.poll() is None else None)

    base_score = score_via_pipe(p, base["code"])
    print(f"[kiss] baseline {base['name']}: {base_score}", flush=True)

    state = {"best_sim": base_score.get("sim_time_us", 1e18),
             "best_code": base["code"], "best_name": base["name"],
             "n_calls": 0, "n_ok": 0}

    def score_candidate(code: str) -> str:
        """Score a Python implementation. Returns JSON with 'ok' and
        'sim_time_us' (lower is better)."""
        r = score_via_pipe(p, code)
        state["n_calls"] += 1
        if r.get("ok"):
            state["n_ok"] += 1
            if r["sim_time_us"] < state["best_sim"]:
                state["best_sim"] = r["sim_time_us"]
                state["best_code"] = code
                state["best_name"] = f"kiss_{state['n_calls']}"
        return json.dumps(r)[:1200]

    prompt = GENERIC_EVO_PROMPT
    prompt = prompt.replace("{current_code}", base["code"])
    prompt = prompt.replace("{current_sim_time}",
                            str(base_score.get("sim_time_us", 0)))
    for k in ("{current_num_permutes}", "{current_num_gathers}",
              "{current_local_ops}", "{history}", "{efa_bandwidth}",
              "{efa_latency}", "{builtin_ag_slice_cat}",
              "{builtin_permute_ring}"):
        prompt = prompt.replace(k, "0" if k.endswith("}") else "")
    prompt = prompt.replace("{world_size}", "224")
    prompt = prompt.replace("{num_devices}", "112")
    prompt = prompt.replace("{cores_per_device}", "2")
    prompt = prompt.replace("{num_nodes}", "7")
    prompt = prompt.replace("{ranks_per_node}", "32")
    prompt += ("\n\nUse the score_candidate(code: str) tool to evaluate any "
               "new implementation. Aim to minimize sim_time_us. Call finish "
               "when you cannot improve further.")

    agent = KISSAgent(f"fair-{args.problem}")
    t0 = time.time()
    try:
        agent.run(model_name="claude-sonnet-4-5-20250929",
                  prompt_template=prompt, tools=[score_candidate],
                  is_agentic=True, max_steps=args.max_steps,
                  max_budget=args.max_budget, verbose=False)
    except SystemExit:
        pass
    except Exception as e:
        print(f"[kiss] agent error: {e}")
    wall = time.time() - t0

    try:
        p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
    except Exception:
        pass

    summary = {"pipeline": "sorcar", "problem": args.problem,
               "gate": args.gate, "wall_seconds": wall,
               "baseline_sim_time_us": base_score.get("sim_time_us"),
               "best_sim_time_us": state["best_sim"],
               "best_name": state["best_name"],
               "n_score_calls": state["n_calls"], "n_ok": state["n_ok"]}
    with open(os.path.join(args.output_dir, "best_code.py"), "w") as f:
        f.write(state["best_code"])
    with open(os.path.join(args.output_dir, "kiss_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
