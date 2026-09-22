"""Faithful pre-screen for r53 (MAX/MIN-semiring telescoping, fam-4 candidate).

baseline gate-pass+sim; IDEAL fold (one AR(MAX/MIN) of x+beta then + (D-1)*max/min(beta))
gate-pass+sim (headroom); NAIVE drop-beta guess (D AR(MAX) with no beta -> should be WRONG,
fail the gate, proving the max-plus fold is non-obvious).
"""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32


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


def beta_expr(gamma):
    return f"({gamma} * (((rank * 37) % world_size) - world_size/2.0) / world_size)"


def fold_maxplus(name, depth, gamma, is_max):
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    # net local constant = (D-1)*extremum_r(beta[r]); compute extremum over all ranks locally.
    ext = "max" if is_max else "min"
    return (sig(name) +
            f"    W = world_size; G = {gamma}\n"
            f"    beta = {beta_expr(gamma)}\n"
            f"    m = xm.all_reduce({red}, x + beta)\n"
            f"    allbeta = [G*(((rr*37) % W) - W/2.0)/W for rr in range(W)]\n"
            f"    C = {ext}(allbeta)\n"
            f"    return m + ({depth} - 1) * C\n")


def naive_dropbeta(name, depth, is_max):
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    # WRONG guess: forget beta entirely, just AR(MAX,x) once
    return sig(name) + f"    return xm.all_reduce({red}, x)\n"


PROBS = {
    "r53_maxshift_d8_n1024": (8, 2.0, True),
    "r53_maxshift_d6_n1024": (6, 2.0, True),
    "r53_maxshift_d4_n1024": (4, 2.0, True),
    "r53_maxshift_d8_g1": (8, 1.0, True),
    "r53_maxshift_d8_g4": (8, 4.0, True),
    "r53_minshift_d8_n1024": (8, 2.0, False),
    "r53_minshift_d6_n1024": (6, 2.0, False),
    "r53_maxshift_d8_res": (8, 2.0, True),
    "r53_minshift_d8_res": (8, 2.0, False),
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
        capture_output=True, text=True, timeout=90, env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-160:])


def main():
    print(f"=== r53 MAX/MIN-semiring pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, (depth, gamma, is_max) in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, fold_maxplus(name, depth, gamma, is_max))
            nv = score(p, naive_dropbeta(name, depth, is_max))
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
            note = "BASE_FAIL:" + str(b.get("error"))[:90]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:90]
        print(f"{name:24s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f'%hr) if hr else '-':>6} naive_guess_ok={nv.get('ok')} "
              f"{'(TRAP: guess fails)' if not nv.get('ok') else '(guess PASSES)'} {note}")
    print("\n=== distinctness (md5 @ seed 12345) ===")
    hashes = {}
    for name, *_ in rows:
        h = ref_hash(name); hashes[name] = h
        print(f"  {name:24s} {h[:16]}")
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE")
    print("\n=== VIABLE (base+fold gate-pass, hr>=1.05, naive fails) ===")
    viable = []
    for name, b_ok, b_sim, f_ok, f_sim, hr, nv_ok in rows:
        ok = bool(b_ok and f_ok and hr and hr >= 1.05)
        if ok:
            viable.append(name)
        print(f"  {name:24s} {'VIABLE' if ok else 'no    '} hr={hr} naive_fails={not nv_ok}")
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r53.json", "w"), indent=2)


if __name__ == "__main__":
    main()
