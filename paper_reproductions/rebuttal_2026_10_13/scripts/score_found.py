#!/usr/bin/env python3
"""Score the programs found by the leak-free generic search under a given simulator arm and compare with the
baseline and the deployed (LLM) strategy for each problem.  Run ON master:
  cd /home/ubuntu/OverlayCCL && PYTHONPATH=. $VENV/bin/python /home/ubuntu/ap7/score_found.py [TABLE=hw_probe_measured.json]
"""
import json, os, sys, ast
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO); sys.path.insert(0, "/home/ubuntu/ap7")
os.environ.setdefault("MANIFEST", "/home/ubuntu/ap7/manifest_v2.json")
import score_v3 as SV            # importing runs the pool scoring too (cheap, CPU) and gives ARMS/code_for
from search.problems import get_problem
from search.correctness_test import benchmark_xla_candidate_generic
from experiments.run_search import _problem_train_scale_multiplier
G = json.load(open(os.environ.get("FOUND", "/home/ubuntu/sweep_results/generic_search_default.json")))
man = SV.man
DEPLOYED = {"alltoallv": "paper_runtime", "uniform_a2a": "paper_runtime", "ring_kv": "paper_runtime", "dxe": "paper_runtime",
            "pp_send_recv": "paper_runtime", "tp_mlp": "paper_runtime", "fsdp_prefetch": "paper_runtime", "llama_block_ar": "paper_runtime"}
def score(prob, code, arm):
    ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
    kw = dict(SV.ARMS[arm]["kw"]); kw["training_scale_bytes_multiplier"] = _problem_train_scale_multiplier(prob)
    b = benchmark_xla_candidate_generic(prob, fn, SV.ARMS[arm]["topo"], SV.send, SV.WORLD, num_nodes=SV.NODES, unsupported_primitives=["all_to_all"], **kw)
    return None if "error" in b else round(b["sim_time_us"], 1)
out = {}
for arm in SV.ARMS:
    print(f"\n=== arm {arm} ===")
    print(f"{'problem':15s} {'baseline':>10s} " + " ".join(f"{m:>10s}" for m in ("random", "ga", "beam", "rules")) + f" {'deployed':>10s}  deployed-cand")
    for pname, r in G.items():
        prob = get_problem(pname)
        base_code = r.get("start_code") or None
        row = {}
        for m in ("random", "ga", "beam", "rules"):
            c = r[m].get("code"); row[m] = score(prob, c, arm) if c else None
        # baseline + deployed from the pool manifest (kind=file entries whose cand name marks the paper's deployed runtime)
        dep = [c for c in man if c["problem"] == pname and c["kind"] == "file"]
        dep_scores = {c["cand"]: SV.out[arm].get(f"{pname}__{c['cand']}") if hasattr(SV, "out") else None for c in dep}
        bl = next((c for c in man if c["problem"] == pname and c["kind"] == "harness_baseline"), None)
        base = SV.out[arm].get(f"{pname}__{bl['cand']}") if (bl and hasattr(SV, "out")) else None
        out.setdefault(arm, {})[pname] = dict(found=row, baseline=base, deployed=dep_scores)
        best_dep = min(((v, k) for k, v in dep_scores.items() if v is not None), default=(None, "-"))
        print(f"{pname:15s} {str(base):>10s} " + " ".join(f"{str(row[m]):>10s}" for m in ("random", "ga", "beam", "rules")) + f" {str(best_dep[0]):>10s}  {best_dep[1][:40]}")
json.dump(out, open("/home/ubuntu/sweep_v2/found_scores.json", "w"), indent=1)
