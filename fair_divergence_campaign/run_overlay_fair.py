"""OverlayCCL (strategy-enumerate) driver against the FAIR symmetric scorer.

Faithful to search/strategy_enumerate_phase3.py's search shape:
  1 enumerate (K=5 structural strategies)
  -> implement each (K calls)
  -> sim-rank valid ones, take top-2
  -> refine each top-2 for R rounds with sim-score + dominant-term feedback.

The ONLY difference from the shipped strat pipeline: correctness+sim scoring
goes through score_service_fair.py (GATE_MODE-controlled), the SAME service
kiss uses here. So both pipelines face the identical gate. Uses the verbatim
ENUMERATION/IMPLEMENT/REFINE prompts and _parse_strategies/_extract_code.
"""
import argparse
import json
import os
import subprocess
import sys

ACC = os.environ.get("ACC_REPO", "/private/tmp/acc_verify")
sys.path.insert(0, ACC)
SCORE_PY = "/private/tmp/fair_diverge/score_service_fair.py"
PY = sys.executable
MODEL = "sonnet"

from search.strategy_enumerate_phase3 import (  # noqa: E402
    ENUMERATION_PROMPT, IMPLEMENT_PROMPT, REFINE_PROMPT_HEADER,
    _parse_strategies, _extract_code,
)
from search._anthropic_route import _invoke_anthropic  # noqa: E402


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


def score(p, code):
    p.stdin.write(json.dumps({"code": code}) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    return json.loads(line) if line else {"ok": False, "error": "EOF"}


def get_meta(problem):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}');"
         "print(json.dumps({'signature': p.signature,"
         "'signature_doc': p.signature_doc,"
         "'evolved_fn_name': p.evolved_fn_name,"
         "'display_name': p.display_name,"
         "'hints': getattr(p, 'optimization_hints', ''),"
         "'templates': p.builtin_templates}))"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": ACC})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--pattern", default="moe")
    ap.add_argument("--num-nodes", type=int, default=7)
    ap.add_argument("--gate", default="fp32", choices=["fp32", "fp32_bf16"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    meta = get_meta(args.problem)
    world = args.num_nodes * 32
    ref_block = "\n\n".join(f"### {n}:\n```python\n{c}\n```"
                            for n, c in meta["templates"].items())
    hints = meta.get("hints") or "(use only xm collectives; minimize dispatches)"

    p = start_scorer(args.problem, args.pattern, args.num_nodes, args.gate)
    base_code = next(iter(meta["templates"].values()))
    base_sc = score(p, base_code)
    base = base_sc.get("sim_time_us")

    log = {"pipeline": "overlay", "problem": args.problem, "gate": args.gate,
           "baseline_sim": base, "strategies": [], "candidates": [],
           "n_llm_calls": 0, "final_sim": None, "final_code": base_code}

    # Stage 1: enumerate K strategies.
    enum_prompt = ENUMERATION_PROMPT.format(
        world_size=world, display_name=meta["display_name"],
        signature=meta["signature"], signature_doc=meta["signature_doc"],
        reference_implementations=ref_block, optimization_hints=hints, k=args.k)
    enum_resp = _invoke_anthropic(enum_prompt, model=MODEL, temperature=0.8,
                                  max_tokens=4000)
    log["n_llm_calls"] += 1
    strategies = _parse_strategies(enum_resp, args.k)
    log["strategies"] = [{"name": n, "desc": d} for n, d in strategies]

    # Stage 2: implement each strategy, score under the fair gate.
    cands = []
    for name, desc in strategies:
        ip = IMPLEMENT_PROMPT.format(
            display_name=meta["display_name"], strategy_name=name,
            strategy_description=desc, signature=meta["signature"],
            signature_doc=meta["signature_doc"], optimization_hints=hints,
            reference_implementations=ref_block,
            evolved_fn_name=meta["evolved_fn_name"])
        resp = _invoke_anthropic(ip, model=MODEL, temperature=1.0, max_tokens=4000)
        log["n_llm_calls"] += 1
        code = _extract_code(resp)
        sc = score(p, code) if code else {"ok": False, "error": "no code"}
        sim = sc.get("sim_time_us") if sc.get("ok") else None
        cands.append({"name": name, "code": code, "sim": sim,
                      "ok": bool(sc.get("ok"))})
        log["candidates"].append({"stage": "impl", "name": name, "sim": sim,
                                  "ok": bool(sc.get("ok")),
                                  "err": None if sc.get("ok") else str(sc.get("error"))[:120]})

    # Stage 3: refine top-2 valid by sim.
    valid = [c for c in cands if c["ok"] and c["sim"] is not None]
    valid.sort(key=lambda c: c["sim"])
    top2 = valid[:2]
    for c in top2:
        cur_code, cur_sim = c["code"], c["sim"]
        for rnd in range(args.rounds):
            rp = REFINE_PROMPT_HEADER.format(
                display_name=meta["display_name"], current_code=cur_code,
                sim_us=cur_sim, num_collective_permute="?", num_all_gather="?",
                num_all_reduce="?", local_ops="?",
                dominant_term="collective dispatch / local op count",
                breakdown_text="(reduce the dominant cost term)", history="")
            resp = _invoke_anthropic(rp, model=MODEL, temperature=1.0, max_tokens=4000)
            log["n_llm_calls"] += 1
            nc = _extract_code(resp)
            if not nc:
                continue
            sc = score(p, nc)
            if sc.get("ok") and sc.get("sim_time_us") is not None:
                ns = sc["sim_time_us"]
                log["candidates"].append({"stage": f"refine:{c['name']}",
                                          "round": rnd + 1, "sim": ns})
                if ns < cur_sim:
                    cur_code, cur_sim = nc, ns
        c["refined_sim"] = cur_sim
        c["refined_code"] = cur_code

    # Pick overall best VALID candidate; if none valid, fall back to baseline.
    finals = [(c.get("refined_sim", c["sim"]), c.get("refined_code", c["code"]))
              for c in top2]
    if finals:
        finals.sort(key=lambda t: t[0])
        log["final_sim"], log["final_code"] = finals[0]
    else:
        # No valid candidate survived the gate -> ship baseline (real fallback).
        log["final_sim"], log["final_code"] = base, base_code
        log["fell_back_to_baseline"] = True

    try:
        p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
    except Exception:
        pass

    with open(os.path.join(args.output_dir, "overlay.json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(args.output_dir, "best_code.py"), "w") as f:
        f.write(log["final_code"])
    print(json.dumps({"pipeline": "overlay", "problem": args.problem,
                      "gate": args.gate, "baseline": base,
                      "n_strategies": len(strategies),
                      "n_llm_calls": log["n_llm_calls"],
                      "impl_sims": [c["sim"] for c in cands],
                      "final_sim": log["final_sim"],
                      "fell_back": log.get("fell_back_to_baseline", False)},
                     indent=2))


if __name__ == "__main__":
    main()
