"""Faithful headroom + distinctness pre-screen for the r47 family-3 battery."""
import json, os, subprocess, sys, hashlib

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32
NBLOCK = 8
TOPK = 3


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
    return json.loads(r.stdout.strip().splitlines()[-1])[problem]


def dd_fold(name, part, kind):
    if kind == "meanabs":
        g = ["    f = [1.0 + s[b*S:(b+1)*S].mean().abs() for b in range(B)]"]
    elif kind == "relu":
        g = ["    f = []",
             "    for b in range(B):",
             "        mb = s[b*S:(b+1)*S].mean()",
             "        f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    elif kind == "norm":
        g = ["    nb = [s[b*S:(b+1)*S].norm() for b in range(B)]",
             "    tot = sum(nb) + 1e-6",
             "    f = [1.0 + nb[b]/tot for b in range(B)]"]
    elif kind == "topk":
        g = ["    ma = [s[b*S:(b+1)*S].mean().abs() for b in range(B)]",
             "    order = sorted(range(B), key=lambda b: float(ma[b]), reverse=True)",
             f"    sel = set(order[:{TOPK}])",
             "    f = [2.0 if b in sel else 1.0 for b in range(B)]"]
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"] + g + [
         "    out = s.clone()",
         "    for b in range(B):",
         "        out[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
         "    return out"]
    return "\n".join(L) + "\n"


def xc_fold(name, rounds):
    return "\n".join([
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        f"    return s * (world_size ** {rounds - 1})"]) + "\n"


PROBS = [
    ("r47_dd_meanabs_d8",  lambda n: dd_fold(n, 256, "meanabs")),
    ("r47_dd_relu_d8",     lambda n: dd_fold(n, 256, "relu")),
    ("r47_dd_norm_d8",     lambda n: dd_fold(n, 256, "norm")),
    ("r47_dd_topk_d8",     lambda n: dd_fold(n, 256, "topk")),
    ("r47_dd_meanabs_d10", lambda n: dd_fold(n, 256, "meanabs")),
    ("r47_dd_meanabs_d7",  lambda n: dd_fold(n, 256, "meanabs")),
    ("r47_dd_topk_d10",    lambda n: dd_fold(n, 256, "topk")),
    ("r47_dd_meanabs_d8_p384", lambda n: dd_fold(n, 384, "meanabs")),
    ("r47_dd_topk_d8_p384",    lambda n: dd_fold(n, 384, "topk")),
    ("r47_dd_topk_d8_res",  lambda n: dd_fold(n, 256, "topk")),
    ("r47_dd_norm_d8_res",  lambda n: dd_fold(n, 256, "norm")),
    ("r47_xc_r2", lambda n: xc_fold(n, 2)),
    ("r47_xc_r3", lambda n: xc_fold(n, 3)),
    ("r47_xc_r4", lambda n: xc_fold(n, 4)),
]


def ref_hash(problem):
    """Distinctness: hash the rank-0 expected output at a fixed seed."""
    r = subprocess.run(
        [PY, "-c",
         "import sys, json, hashlib; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}');"
         f"tc = p.generate_test_case({WORLD}, seed=12345);"
         "e = tc['expected'][0];"
         "print(hashlib.md5(e.numpy().tobytes()).hexdigest())"],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-200:])


def main():
    print(f"=== r47 family-3 pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, foldf in PROBS:
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:160]}"); rows.append((name, None, None, None, "scorer_fail")); continue
        try:
            b = score(p, get_baseline(name)); f = score(p, foldf(name))
        finally:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
            except Exception:
                pass
            p.terminate()
        b_ok, f_ok = b.get("ok"), f.get("ok")
        b_sim, f_sim = b.get("sim_time_us"), f.get("sim_time_us")
        hr = (b_sim / f_sim) if (b_ok and f_ok and f_sim) else None
        note = "" if (b_ok and f_ok) else ("BASE_FAIL:" + str(b.get("error"))[:110] if not b_ok else "FOLD_FAIL:" + str(f.get("error"))[:110])
        rows.append((name, b_sim, f_sim, hr, note))
        print(f"{name:26s} base={str(b_sim)[:9]:>9s} fold={str(f_sim)[:9]:>9s} hr={('%.3f'%hr) if hr else '-':>6s}  {note}")
    print("\n=== distinctness (rank-0 expected md5 @ seed 12345) ===")
    hashes = {}
    for name, *_ in rows:
        h = ref_hash(name); hashes[name] = h
        print(f"  {name:26s} {h[:16]}")
    # group collisions
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("\n  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE (all distinct)")
    print("\n=== verdict (VIABLE = gate-pass both + headroom>=1.05) ===")
    for name, b_sim, f_sim, hr, note in rows:
        print(f"  {name:26s} {'VIABLE' if (hr and hr>=1.05) else 'no    '} hr={hr}  {note}")
    json.dump({"rows": [{"prob": r[0], "base_sim": r[1], "fold_sim": r[2], "headroom": r[3], "note": r[4]} for r in rows],
               "hashes": hashes}, open(f"{FD}/prescreen_r47.json", "w"), indent=2)


if __name__ == "__main__":
    main()
