"""Faithful pre-screen for r58 (r56-expansion battery)."""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32
NG = 4


def start_scorer(problem):
    env = {**os.environ, "PYTHONPATH": ACC, "ACC_REPO": ACC,
           "SCORE_PROBLEM": problem, "SCORE_PATTERN": "moe",
           "SCORE_NUM_NODES": str(NODES), "GATE_MODE": "fp32", "ANTHROPIC_API_KEY": "dummy"}
    p = subprocess.Popen([PY, SCORE_PY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env, text=True, bufsize=1)
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


def sig(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            f"                 cores_per_device, xm, torch, num_nodes=1):\n")


A_EXPR = "1.0 + 0.5*((r*13) % 5)/5.0"
B_EXPR = "1.0 + 0.25*((r*7) % 9)/9.0"


def grp_expr(strided):
    return ("[[r for r in range(W) if r % NG == j] for j in range(NG)]" if strided else
            "[list(range(j*(W//NG), (j+1)*(W//NG))) for j in range(NG)]")


def gid_expr(strided):
    return "(rank % NG)" if strided else "(rank // (W//NG))"


def fold_alt(name, depth, strided, swap):
    fa, fb = (B_EXPR, A_EXPR) if swap else (A_EXPR, B_EXPR)
    return (sig(name) +
            f"    W = world_size; NG = {NG}; D = {depth}\n"
            f"    groups = {grp_expr(strided)}\n"
            f"    fa = lambda r: {fa}\n"
            f"    fb = lambda r: {fb}\n"
            "    m1 = xm.all_reduce(xm.REDUCE_SUM, fa(rank) * x, groups=groups)\n"
            "    u = xm.all_reduce(xm.REDUCE_SUM, fb(rank) * m1)\n"
            "    A = [sum(fa(r) for r in G) for G in groups]\n"
            "    B = [sum(fb(r) for r in G) for G in groups]\n"
            "    BA = sum(B[k]*A[k] for k in range(NG))\n"
            "    return u * (BA ** ((D - 2)//2))\n")


def fold_grponly(name, depth, strided):
    return (sig(name) +
            f"    W = world_size; NG = {NG}; D = {depth}\n"
            f"    groups = {grp_expr(strided)}\n"
            f"    fa = lambda r: {A_EXPR}\n"
            "    m1 = xm.all_reduce(xm.REDUCE_SUM, fa(rank) * x, groups=groups)\n"
            f"    g = {gid_expr(strided)}\n"
            "    Ag = sum(fa(r) for r in groups[g])\n"
            "    return m1 * (Ag ** (D - 1))\n")


def naive(name, depth, grponly, swap):
    fa, fb = (B_EXPR, A_EXPR) if swap else (A_EXPR, B_EXPR)
    body = (sig(name) +
            f"    W = world_size; D = {depth}\n"
            f"    a = (lambda r: {fa})(rank)\n"
            f"    b = (lambda r: {fb})(rank)\n"
            "    cur = x\n"
            "    for t in range(D):\n")
    if grponly:
        body += "        cur = xm.all_reduce(xm.REDUCE_SUM, a * cur)\n"
    else:
        body += "        cur = xm.all_reduce(xm.REDUCE_SUM, (a if t % 2 == 0 else b) * cur)\n"
    return body + "    return cur\n"


# name -> (depth, strided, grponly, swap)
PROBS = {
    "r58_grpcont_d8_n1024": (8, False, False, False),
    "r58_grponly_d8_n1024": (8, True, True, False),
    "r58_grponly_d6_n1024": (6, True, True, False),
    "r58_grponly_d8_cont": (8, False, True, False),
    "r58_hier_d10_n1024": (10, True, False, False),
    "r58_hier_d8_n8192": (8, True, False, False),
    "r58_gswap_d8_n1024": (8, True, False, True),
    "r58_grponly_d8_res": (8, True, True, False),
}


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
        capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-160:])


def main():
    print(f"=== r58 expansion pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, (depth, strided, grponly, swap) in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            fcode = (fold_grponly(name, depth, strided) if grponly
                     else fold_alt(name, depth, strided, swap))
            f = score(p, fcode)
            nv = score(p, naive(name, depth, grponly, swap))
        finally:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
            except Exception:
                pass
            p.terminate()
        b_ok, f_ok = b.get("ok"), f.get("ok")
        b_sim, f_sim = b.get("sim_time_us"), f.get("sim_time_us")
        hr = (b_sim / f_sim) if (b_ok and f_ok and f_sim) else None
        rows.append((name, b_ok, b_sim, f_ok, f_sim, hr, nv.get("ok")))
        note = ""
        if not b_ok:
            note = "BASE_FAIL:" + str(b.get("error"))[:100]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:100]
        print(f"{name:22s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f'%hr) if hr else '-':>6} naive_ok={nv.get('ok')} "
              f"{'(TRAP: guess fails)' if not nv.get('ok') else '(guess PASSES)'} {note}")
    print("\n=== distinctness (md5 @ seed 12345) ===")
    hashes = {}
    for r in rows:
        name = r[0]
        h = ref_hash(name); hashes[name] = h
        print(f"  {name:22s} {h[:16]}")
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE")
    viable = [r[0] for r in rows if (r[1] and r[3] and r[5] and r[5] >= 1.05)]
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r58.json", "w"), indent=2)


if __name__ == "__main__":
    main()
