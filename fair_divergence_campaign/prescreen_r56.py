"""Faithful pre-screen for r56 (hierarchical group-wise AR telescoping, fam-4 candidate).

baseline gate-pass + sim; IDEAL fold gate-pass + sim (headroom); NAIVE guess (drop the groups
structure, D global ARs of the same scalars -> should be WRONG, fail the gate).

Fold derivation: stage 0 (group) gives m1_j = sum_{r in G_j} a[r] x_r (this rank sees its own
group's m1). Stage 1 (global) gives u = sum_k B_k m1_k where B_k = sum_{r in G_k} b[r]. Every
later stage acts on group-values by LOCAL constants: group stage v_j = A_j*u (A_j = sum_{r in G_j}
a[r]); global stage u' = (sum_k B_k A_k) * u. For even depth D the final is
u * (sum_k B_k A_k)^((D-2)/2) -- so the fold is exactly 2 collectives + one local scalar multiply.
"""
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


def fold(name, depth):
    # 2 collectives + local closed form (see module docstring).
    return (sig(name) +
            f"    W = world_size; NG = {NG}; D = {depth}\n"
            "    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]\n"
            "    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0\n"
            "    m1 = xm.all_reduce(xm.REDUCE_SUM, a * x, groups=groups)\n"
            "    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0\n"
            "    u = xm.all_reduce(xm.REDUCE_SUM, b * m1)\n"
            "    A = [sum(1.0 + 0.5*((r*13) % 5)/5.0 for r in range(W) if r % NG == j)\n"
            "         for j in range(NG)]\n"
            "    B = [sum(1.0 + 0.25*((r*7) % 9)/9.0 for r in range(W) if r % NG == j)\n"
            "         for j in range(NG)]\n"
            "    BA = sum(B[k]*A[k] for k in range(NG))\n"
            "    return u * (BA ** ((D - 2)//2))\n")


def naive(name, depth):
    # WRONG: ignore groups, all-global chain
    return (sig(name) +
            f"    D = {depth}\n"
            "    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0\n"
            "    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0\n"
            "    cur = x\n"
            "    for t in range(D):\n"
            "        cur = xm.all_reduce(xm.REDUCE_SUM, (a if t % 2 == 0 else b) * cur)\n"
            "    return cur\n")


PROBS = {
    "r56_hier_d8_n1024": 8,
    "r56_hier_d6_n1024": 6,
    "r56_hier_d4_n1024": 4,
    "r56_hier_d8_n4096": 8,
    "r56_hier_d8_res": 8,
    "r56_hier_d6_res": 6,
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
    print(f"=== r56 hierarchical group-AR pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, depth in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, fold(name, depth))
            nv = score(p, naive(name, depth))
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
        print(f"{name:20s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f'%hr) if hr else '-':>6} naive_ok={nv.get('ok')} "
              f"{'(TRAP: guess fails)' if not nv.get('ok') else '(guess PASSES)'} {note}")
    print("\n=== distinctness (md5 @ seed 12345) ===")
    hashes = {}
    for r in rows:
        name = r[0]
        h = ref_hash(name); hashes[name] = h
        print(f"  {name:20s} {h[:16]}")
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE")
    viable = [r[0] for r in rows if (r[1] and r[3] and r[5] and r[5] >= 1.05)]
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r56.json", "w"), indent=2)


if __name__ == "__main__":
    main()
