"""Faithful headroom pre-screen for the r46 family-3 candidates (axes B/C).

For each problem: start the REAL fair scorer (score_service_fair.py) for that
problem/pattern/nodes, then score (1) the builtin baseline template and (2) the
ideal FOLD. Report gate-pass + sim + headroom = baseline_sim / fold_sim. This is
the same gate + cost model the cloud run will use, so a problem that doesn't show
gate-pass baseline + real headroom here won't diverge and shouldn't cost tokens.
"""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32  # 224

NBLOCK = 8


def start_scorer(problem, pattern="moe", gate="fp32"):
    env = {**os.environ, "PYTHONPATH": ACC, "ACC_REPO": ACC,
           "SCORE_PROBLEM": problem, "SCORE_PATTERN": pattern,
           "SCORE_NUM_NODES": str(NODES), "GATE_MODE": gate,
           "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "dummy")}
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


def get_baseline(problem):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}');"
         "print(json.dumps(p.builtin_templates))"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": ACC})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])
    tpl = json.loads(r.stdout.strip().splitlines()[-1])
    return tpl[problem]


# ---- ideal folds ---------------------------------------------------------
def fold_datadiag(name, part):
    return "\n".join([
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        "    out = s.clone()",
        "    for b in range(B):",
        "        m = s[b*S:(b+1)*S].mean().abs()",
        "        out[b*S:(b+1)*S] = s[b*S:(b+1)*S] * (1.0 + m)",
        "    return out",
    ]) + "\n"


def fold_xcoll(name, part, rounds):
    return "\n".join([
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        f"    return s * (world_size ** {rounds - 1})",
    ]) + "\n"


PROBS = [
    ("r46_datadiag_d8_count8", lambda n: fold_datadiag(n, 256)),
    ("r46_datadiag_d6_count8", lambda n: fold_datadiag(n, 256)),
    ("r46_datadiag_d8_res",    lambda n: fold_datadiag(n, 256)),
    ("r46_datadiag_d8_big",    lambda n: fold_datadiag(n, 1024)),
    ("r46_xcoll_r3_count8",    lambda n: fold_xcoll(n, 256, 3)),
    ("r46_xcoll_r2_count8",    lambda n: fold_xcoll(n, 256, 2)),
    ("r46_xcoll_r3_res",       lambda n: fold_xcoll(n, 256, 3)),
]


def main():
    print(f"=== r46 family-3 pre-screen @ W={WORLD} (nodes={NODES}), gate=fp32 ===")
    rows = []
    for name, foldf in PROBS:
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:200]}")
            rows.append((name, None, None, None, "scorer_start_fail"))
            continue
        try:
            base_code = get_baseline(name)
            b = score(p, base_code)
            f = score(p, foldf(name))
        finally:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
            except Exception:
                pass
            p.terminate()
        b_ok = b.get("ok"); f_ok = f.get("ok")
        b_sim = b.get("sim_time_us"); f_sim = f.get("sim_time_us")
        hr = (b_sim / f_sim) if (b_ok and f_ok and f_sim) else None
        note = ""
        if not b_ok:
            note = "BASE_FAIL:" + str(b.get("error"))[:120]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:120]
        rows.append((name, b_sim, f_sim, hr, note))
        hr_s = f"{hr:.3f}" if hr else "-"
        print(f"{name:28s} base={str(b_sim)[:9]:>9s} fold={str(f_sim)[:9]:>9s} "
              f"headroom={hr_s:>6s}  {note}")
    print("\n=== verdict ===")
    for name, b_sim, f_sim, hr, note in rows:
        viable = hr is not None and hr >= 1.05
        print(f"  {name:28s} {'VIABLE' if viable else 'no    '} "
              f"headroom={hr if hr else '-'}  {note}")
    json.dump([{"prob": r[0], "base_sim": r[1], "fold_sim": r[2],
                "headroom": r[3], "note": r[4]} for r in rows],
              open(f"{FD}/prescreen_r46.json", "w"), indent=2)


if __name__ == "__main__":
    main()
