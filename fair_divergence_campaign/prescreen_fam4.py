"""Offline pre-screen for the FAMILY-4 candidate batteries:
  direction A -> r62 (positional-shift: out[b] = a[b]*s[(b+k)%B])
  direction B -> r63 (global-scalar coupling: out = s + beta*stat(s))

No Bedrock. For each candidate: (1) baseline (the deep telescoping chain) must PASS the
fp32 gate at world_sizes [4,8,224] -- this doubles as a correctness check of the telescoping;
(2) an ideal 1-AR fold must pass and give headroom >= 1.05; (3) naive plain AR must FAIL
(the trap holds); (4) STRICT non-duplication -- md5 of each candidate's reference at W=224
vs EVERY already-registered problem.
"""
import json
import os
import subprocess
import sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32
NBLOCK = 8

# ---- candidate tables (must mirror the register_all() in r62.py / r63.py) ----
_A = {"lin": "1.0 + 0.5*(b % 3)", "alt": "1.0 + 0.25*(b % 4)"}
R62 = {  # name -> (part, k, a_kind)
    "r62_shift_k1_p512_d8":     (512,  1, "lin"),
    "r62_shift_k1_p1024_d8":    (1024, 1, "lin"),
    "r62_shift_k1_p2048_d8":    (2048, 1, "lin"),
    "r62_shift_k2_p1024_d8":    (1024, 2, "lin"),
    "r62_shift_k2_p2048_d8":    (2048, 2, "lin"),
    "r62_shift_k3_p1024_d8":    (1024, 3, "lin"),
    "r62_shift_k3_p2048_d8":    (2048, 3, "lin"),
    "r62_shift_k1_alt_p2048_d8": (2048, 1, "alt"),
    "r62_shift_k2_alt_p2048_d8": (2048, 2, "alt"),
}
R63 = {  # name -> (part, stat, beta)
    "r63_gmean_b0p5_p2048_d8": (2048, "mean", 0.5),
    "r63_gmean_b1p0_p2048_d8": (2048, "mean", 1.0),
    "r63_gmean_b2p0_p2048_d8": (2048, "mean", 2.0),
    "r63_gmean_b1p0_p1024_d8": (1024, "mean", 1.0),
    "r63_gmean_b1p0_p512_d8":  (512,  "mean", 1.0),
    "r63_gsum_b0p01_p2048_d8": (2048, "sum", 0.01),
    "r63_gsum_b0p02_p2048_d8": (2048, "sum", 0.02),
}
NEW = list(R62) + list(R63)


def _sig(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            f"                 cores_per_device, xm, torch, num_nodes=1):\n")


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


def ref_hash(problem, world=WORLD):
    r = subprocess.run(
        [PY, "-c",
         "import sys, json, hashlib; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import get_problem;"
         f"p = get_problem('{problem}');"
         f"tc = p.generate_test_case({world}, seed=12345);"
         "e = tc['expected'][0];"
         "print(hashlib.md5(e.numpy().tobytes()).hexdigest())"],
        capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-160:])


def fold_r62(name):
    part, k, akind = R62[name]
    L = [_sig(name).rstrip("\n"),
         f"    S = {part}; B = {NBLOCK}; K = {k}",
         f"    a = [{_A[akind]} for b in range(B)]",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         "    out = s.clone()",
         "    for b in range(B):",
         "        src = (b + K) % B",
         "        out[b*S:(b+1)*S] = a[b] * s[src*S:(src+1)*S]",
         "    return out"]
    return "\n".join(L) + "\n"


def fold_r63(name):
    part, stat, beta = R63[name]
    g = "s.mean()" if stat == "mean" else "s.sum()"
    L = [_sig(name).rstrip("\n"),
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         f"    return s + {beta} * ({g})"]
    return "\n".join(L) + "\n"


def naive_plain(name):
    return _sig(name) + "    return xm.all_reduce(xm.REDUCE_SUM, x)\n"


def make_fold(name):
    return fold_r62(name) if name in R62 else fold_r63(name)


def all_registered():
    r = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import PROBLEMS; print('\\n'.join(sorted(PROBLEMS)))"],
        capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": ACC})
    return [x for x in r.stdout.strip().splitlines() if x]


def main():
    print(f"=== FAMILY-4 (r62 shift / r63 global-couple) pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name in NEW:
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}")
            rows.append((name, False, None, False, None, None, None)); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, make_fold(name))
            nv = score(p, naive_plain(name))
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
            note = "BASE_FAIL:" + str(b.get("error"))[:140]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:140]
        print(f"{name:28s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f' % hr) if hr else '-':>6} naive_ok={nv.get('ok')} "
              f"{'(TRAP)' if nv.get('ok') is False else '(naive PASSES!)'} {note}")

    print("\n=== STRICT non-duplication: new md5 vs ALL registered ===")
    new_h = {n: ref_hash(n) for n in NEW}
    for n, h in new_h.items():
        print(f"  {n:28s} {h[:16]}")
    existing = [x for x in all_registered() if x not in R62 and x not in R63]
    exist_h = {x: ref_hash(x) for x in existing}
    coll = []
    inv_exist = {}
    for x, h in exist_h.items():
        if not h.startswith("ERR"):
            inv_exist.setdefault(h, []).append(x)
    for n, h in new_h.items():
        if h.startswith("ERR"):
            coll.append((n, "HASH_ERR", h)); continue
        if h in inv_exist:
            coll.append((n, "DUP_EXISTING", inv_exist[h]))
    inv_new = {}
    for n, h in new_h.items():
        if not h.startswith("ERR"):
            inv_new.setdefault(h, []).append(n)
    for h, ns in inv_new.items():
        if len(ns) > 1:
            coll.append((ns[0], "DUP_INTRA", ns[1:]))
    print("  collisions:", coll if coll else "NONE")

    dup_names = {c[0] for c in coll}
    viable = [r[0] for r in rows
              if (r[1] and r[3] and r[6] is False and r[5] and r[5] >= 1.05
                  and r[0] not in dup_names)]
    print("\nVIABLE (headroom>=1.05, trap holds, distinct):", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "new_hashes": new_h, "collisions": coll, "viable": viable},
              open(f"{FD}/prescreen_fam4.json", "w"), indent=2)


if __name__ == "__main__":
    main()
