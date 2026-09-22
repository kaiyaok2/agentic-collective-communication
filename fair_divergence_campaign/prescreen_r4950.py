"""Faithful gate+headroom+distinctness pre-screen for fam-4 (r49) and fam-5 (r50)."""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32
NBLOCK = 8

_FEXPR = {"sum": "t", "sq": "t*t", "abs": "t.abs()",
          "relu": "t.clamp(min=0.0)", "cube": "t*t*t"}


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
    p.stdin.write(json.dumps({"code": code}) + "\n"); p.stdin.flush()
    line = p.stdout.readline()
    return json.loads(line) if line else {"ok": False, "error": "EOF"}


def get_baseline(problem):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}'); print(json.dumps(p.builtin_templates))"],
        capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": ACC})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])
    return json.loads(r.stdout.strip().splitlines()[-1])[problem]


def stat_fold(name, N, keys):
    # fuse: one AR of cat[f0(x),f1(x),...] then split
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         "    t = x",
         f"    big = torch.cat([{', '.join(_FEXPR[k] for k in keys)}])",
         "    r = xm.all_reduce(xm.REDUCE_SUM, big)",
         "    return r"]
    return "\n".join(L) + "\n"


def lin_fold(name, part, K):
    # fuse: precompute W = sum_k w_k (length N), one AR of W*x
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}",
         "    W = torch.zeros(B*S)",
         f"    for k in range({K}):",
         "        for b in range(B):",
         "            W[b*S:(b+1)*S] += 1.0 + 0.1*((k + b) % 5)",
         "    return xm.all_reduce(xm.REDUCE_SUM, W * x)"]
    return "\n".join(L) + "\n"


PROBS = [
    ("r49_stat_k2",  lambda n: stat_fold(n, 512, ["sum", "sq"])),
    ("r49_stat_k3",  lambda n: stat_fold(n, 512, ["sum", "sq", "abs"])),
    ("r49_stat_k4",  lambda n: stat_fold(n, 512, ["sum", "sq", "abs", "relu"])),
    ("r49_stat_k5",  lambda n: stat_fold(n, 512, ["sum", "sq", "abs", "relu", "cube"])),
    ("r49_stat_k3_p256",  lambda n: stat_fold(n, 256, ["sum", "sq", "abs"])),
    ("r49_stat_k3_p1024", lambda n: stat_fold(n, 1024, ["sum", "sq", "abs"])),
    ("r49_stat_k3_res",   lambda n: stat_fold(n, 512, ["sum", "sq", "abs"])),
    ("r49_stat_k4_res",   lambda n: stat_fold(n, 512, ["sum", "sq", "abs", "relu"])),
    ("r50_lin_k4",  lambda n: lin_fold(n, 256, 4)),
    ("r50_lin_k6",  lambda n: lin_fold(n, 256, 6)),
    ("r50_lin_k8",  lambda n: lin_fold(n, 256, 8)),
    ("r50_lin_k10", lambda n: lin_fold(n, 256, 10)),
    ("r50_lin_k8_p384", lambda n: lin_fold(n, 384, 8)),
    ("r50_lin_k8_p512", lambda n: lin_fold(n, 512, 8)),
    ("r50_lin_k8_res",  lambda n: lin_fold(n, 256, 8)),
    ("r50_lin_k6_res",  lambda n: lin_fold(n, 256, 6)),
]


def ref_hash(problem):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json, hashlib; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}');"
         f"tc = p.generate_test_case({WORLD}, seed=12345);"
         "e = tc['expected'][0];"
         "print(hashlib.md5(e.numpy().tobytes()).hexdigest())"],
        capture_output=True, text=True, timeout=90, env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-160:])


def main():
    print(f"=== fam4(r49)+fam5(r50) pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, foldf in PROBS:
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); rows.append((name, None, None, None, "scorer_fail")); continue
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
        print(f"{name:22s} base={str(b_sim)[:9]:>9s} fold={str(f_sim)[:9]:>9s} hr={('%.3f'%hr) if hr else '-':>6s}  {note}")
    print("\n=== distinctness (md5 @ seed 12345) ===")
    hashes = {}
    for name, *_ in rows:
        h = ref_hash(name); hashes[name] = h
        print(f"  {name:22s} {h[:16]}")
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("\n  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE")
    print("\n=== VIABLE (gate-pass both + hr>=1.05) ===")
    viable = []
    for name, b_sim, f_sim, hr, note in rows:
        ok = bool(hr and hr >= 1.05)
        if ok:
            viable.append(name)
        print(f"  {name:22s} {'VIABLE' if ok else 'no    '} hr={hr}  {note}")
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_sim": r[1], "fold_sim": r[2], "headroom": r[3], "note": r[4]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r4950.json", "w"), indent=2)


if __name__ == "__main__":
    main()
