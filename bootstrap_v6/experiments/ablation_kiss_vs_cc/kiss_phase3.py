"""Phase-3-only kiss-sorcar-react driver.

Runs in the kiss venv (Python 3.13+). Same evolution prompt body as cc-react;
the ReAct controller is kiss's KISSAgent. The `score_candidate(code)` tool
talks to a long-running scorer service in the Neuron venv (Python 3.12) over
pipes; the service is initialised once with cc-react's full Phase 1+2
pipeline (no-LLM defaults), so kiss's candidates are scored against the
exact same cost function cc-react uses.

Environment variables:
  ACC_REPO   Root of the agentic-collective-communication repo. Defaults to
             two directories up from this file (works when this file lives
             at <repo>/experiments/ablation_kiss_vs_cc/).
  NEURON_PY  Path to the Python interpreter of the Neuron venv (the venv
             that has the acc-repo dependencies installed). Defaults to
             /opt/aws_neuronx_venv_pytorch_2_9/bin/python.

Args: --problem <name> --pattern <moe|uniform|...> --output-dir <path>
      [--max-budget 5.0] [--max-steps 30] [--target-score <us>]
"""
import argparse
import atexit
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kiss_token_shim  # noqa  -- patches anthropic SDK for token logging
import sys as _sys
_sys.path.insert(0, '/home/ubuntu')
import kiss_bedrock_shim  # noqa -- monkey-patches AnthropicModel to use Bedrock

ACC = os.environ.get(
    "ACC_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..")))
SCORE_PY = os.path.join(HERE, "score_service_v2.py")
NEURON_PY = os.environ.get(
    "NEURON_PY",
    "/opt/aws_neuronx_venv_pytorch_2_9/bin/python")

from kiss.core.kiss_agent import KISSAgent

GENERIC_EVO_PROMPT = pathlib.Path(
    os.path.join(ACC, "prompts", "generic_evolution_v11.md")).read_text()


def start_scorer(problem, pattern, num_nodes=1):
    env = os.environ.copy()
    env["SCORE_PROBLEM"] = problem
    env["SCORE_PATTERN"] = pattern
    env["SCORE_NUM_NODES"] = str(num_nodes)
    env.setdefault("ACC_REPO", ACC)
    p = subprocess.Popen(
        [NEURON_PY, SCORE_PY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, bufsize=1)
    while True:
        line = p.stderr.readline()
        if not line:
            raise RuntimeError(
                "scorer died before ready: " + (p.stderr.read() or ""))
        sys.stderr.write(f"[scorer init] {line}")
        if "[score_service" in line and "ready" in line:
            break
    return p


def score_via_pipe(p, code, timeout=120):
    p.stdin.write(json.dumps({"code": code}) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        return {"ok": False, "error": "scorer EOF"}
    return json.loads(line)


def get_baseline_code(problem_name):
    r = subprocess.run(
        [NEURON_PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r});"
         "import search.problems;"
         "import search.problems_novel_v4;"
         "import search.problems_novel_v5;"
         "import search.problems_novel_v6;"
         "import search.problems_comm_v7;"
         "import search.problems_challenge_v8;"
         "import search.problems_round17;"
         "import search.problems_round18;"
         "import search.problems_round18b;"
         "import search.problems_round18c;"
         "import search.problems_round20;"
         "import search.problems_round21;"
         "import search.problems_round22;"
         "import search.problems_round23;"
         "import search.problems_round24;"
         "import search.problems_round25;"
         "import search.problems_round26;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem_name}');"
         "tmpls = p.builtin_templates;"
         "k = next(iter(tmpls.keys()));"
         "print(json.dumps({"
         "'name': k, 'code': tmpls[k],"
         "'signature': p.signature, 'signature_doc': p.signature_doc,"
         "'display_name': p.display_name, 'evolved_fn_name': p.evolved_fn_name}))"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"baseline fetch failed: {r.stderr[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-budget", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--llm-model", default="opus", help="opus|sonnet|haiku")
    ap.add_argument("--num-nodes", type=int, default=int(os.environ.get("SCORE_NUM_NODES", "1")))
    ap.add_argument("--target-score", type=float, default=None,
                    help="Stop when score <= target.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = get_baseline_code(args.problem)
    p = start_scorer(args.problem, args.pattern, num_nodes=args.num_nodes)
    atexit.register(lambda: p.terminate() if p.poll() is None else None)

    print('[kiss] scoring baseline...', flush=True)
    import sys
    sys.stdout.flush()
    base_score = score_via_pipe(p, base["code"])
    print(f"[kiss] baseline {base['name']}: {base_score}", flush=True)

    state = {"best_sim": base_score.get("sim_time_us", 1e18),
             "best_code": base["code"],
             "best_name": base["name"],
             "n_calls": 0,
             "early_stop": False}

    def score_candidate(code: str) -> str:
        """Score a Python implementation of the collective. Returns a JSON
        string with 'ok' and 'sim_time_us' (lower is better)."""
        r = score_via_pipe(p, code)
        state["n_calls"] += 1
        if r.get("ok") and r["sim_time_us"] < state["best_sim"]:
            state["best_sim"] = r["sim_time_us"]
            state["best_code"] = code
            state["best_name"] = f"kiss_{state['n_calls']}"
        if (args.target_score is not None and r.get("ok")
                and r["sim_time_us"] <= args.target_score):
            state["early_stop"] = True
        return json.dumps(r)[:1200]

    prompt = GENERIC_EVO_PROMPT
    # NEW: populate signature/signature_doc/evolved_fn_name/display_name
    prompt = prompt.replace("{signature_doc}", base.get("signature_doc", ""))
    prompt = prompt.replace("{signature}", base.get("signature", ""))
    prompt = prompt.replace("{evolved_fn_name}", base.get("evolved_fn_name", ""))
    prompt = prompt.replace("{display_name}", base.get("display_name", args.problem))
    prompt = prompt.replace("{reference_implementations}", "")
    prompt = prompt.replace("{current_code}", base["code"])
    prompt = prompt.replace("{current_sim_time}",
                            str(base_score.get("sim_time_us", 0)))
    for k in ("{current_num_permutes}", "{current_num_gathers}",
              "{current_local_ops}", "{history}", "{efa_bandwidth}",
              "{efa_latency}", "{builtin_ag_slice_cat}",
              "{builtin_permute_ring}"):
        prompt = prompt.replace(k, "0" if k.endswith("}") else "")
    ws = 32 * args.num_nodes
    nd = 16 * args.num_nodes
    prompt = prompt.replace("{world_size}", str(ws))
    prompt = prompt.replace("{num_devices}", str(nd))
    prompt = prompt.replace("{cores_per_device}", "2")
    prompt = prompt.replace("{num_nodes}", str(args.num_nodes))
    prompt = prompt.replace("{ranks_per_node}", "32")
    prompt += ("\n\nUse the score_candidate(code: str) tool to evaluate any "
               "new implementation. Aim to minimize sim_time_us. Call finish "
               "when you cannot improve further.")

    _ref_doc_path = os.path.join(ACC, "prompts", "reference_trainium_details.md")
    def read_reference() -> str:
        """Return the full Trainium reference doc (XLA collectives, unsupported
primitives, sim cost model, worked idioms like torch.tensor([list-comp])
const-fold). Call this whenever you need domain details before writing
candidates."""
        try:
            return open(_ref_doc_path).read()
        except Exception as e:
            return f"ERROR reading reference: {e}"

    agent = KISSAgent(f"phase3-{args.problem}")
    t0 = time.time()
    MODEL_MAP = {"opus": "claude-opus-4-1-20250805", "sonnet": "claude-sonnet-4-5-20250929", "haiku": "claude-haiku-4-5-20251001"}
    try:
        agent.run(model_name=MODEL_MAP.get(args.llm_model, args.llm_model),
                  prompt_template=prompt,
                  tools=[score_candidate, read_reference],
                  is_agentic=True,
                  max_steps=args.max_steps,
                  max_budget=args.max_budget,
                  verbose=False)
    except SystemExit:
        pass
    except Exception as e:
        print(f"[kiss] agent error: {e}")
    wall = time.time() - t0

    try:
        p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        p.stdin.flush()
    except Exception:
        pass

    summary = {
        "problem": args.problem,
        "wall_seconds": wall,
        "baseline_sim_time_us": base_score.get("sim_time_us"),
        "best_sim_time_us": state["best_sim"],
        "best_name": state["best_name"],
        "n_score_calls": state["n_calls"],
        "target_score": args.target_score,
        "hit_target": state["early_stop"],
    }
    with open(os.path.join(args.output_dir, "best_code.py"), "w") as f:
        f.write(state["best_code"])
    with open(os.path.join(args.output_dir, "kiss_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
