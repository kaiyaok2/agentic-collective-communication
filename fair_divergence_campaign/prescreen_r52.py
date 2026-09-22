"""Faithful pre-screen for r52 (collective_permute composition, fam-4 candidate).

For each problem: baseline gate-pass + sim; IDEAL fold gate-pass + sim (headroom);
and the NAIVE obvious-guess check (does Overlay's likely one-shot guess pass or fail?).
A real trap needs: baseline passes, ideal fold passes with headroom>=1.05, AND the
naive guess is either wrong-gate-fail OR structurally hard to reach.
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
           "SCORE_NUM_NODES": str(NODES), "GATE_MODE": "fp32",
           "ANTHROPIC_API_KEY": "dummy"}
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


# ---- ideal folds ----
def fold_rot_net(name, K):
    return (sig(name) +
            f"    W = world_size; K = {K} % W\n"
            "    pairs = [(s, (s + K) % W) for s in range(W)]\n"
            "    return xm.collective_permute(x, pairs=pairs)\n")


def fold_identity(name):
    return sig(name) + "    return x\n"


def fold_permrot(name, shifts):
    # single composed permute: precompute the net holder map, one collective_permute.
    return (sig(name) +
            f"    W = world_size; shifts = {list(shifts)}\n"
            "    PI8 = [3, 0, 5, 7, 1, 6, 2, 4]\n"
            "    pi = list(range(W))\n"
            "    if W % 8 == 0:\n"
            "        for base in range(0, W, 8):\n"
            "            for i in range(8):\n"
            "                pi[base + i] = base + PI8[i]\n"
            "    pinv = [0]*W\n"
            "    for s in range(W):\n"
            "        pinv[pi[s]] = s\n"
            "    holder = list(range(W))\n"
            "    for k in shifts:\n"
            "        holder = [holder[(r - k) % W] for r in range(W)]\n"
            "        holder = [holder[pinv[r]] for r in range(W)]\n"
            # holder[r] = original rank r's tensor now on rank r. Build (src,dst): tensor from
            # holder[r] must arrive at r  => src=holder[r], dst=r.
            "    pairs = [(holder[r], r) for r in range(W)]\n"
            "    return xm.collective_permute(x, pairs=pairs)\n")


# ---- naive WRONG guesses (should FAIL gate if the fold is non-obvious) ----
def naive_rot_first(name, first_k):
    # guess: net rotation == first shift only (common mistake: forget to sum)
    return (sig(name) +
            f"    W = world_size; K = {first_k} % W\n"
            "    pairs = [(s, (s + K) % W) for s in range(W)]\n"
            "    return xm.collective_permute(x, pairs=pairs)\n")


PROBS = {
    "r52_rotnet_d8_n1024": (fold_rot_net, {"K": 3+5+2+7+1+6+4+3}, naive_rot_first, {"first_k": 3}),
    "r52_rotnet_d6_n1024": (fold_rot_net, {"K": 3+5+2+7+1+6}, naive_rot_first, {"first_k": 3}),
    "r52_rotnet_d8_res":   (fold_rot_net, {"K": 3+5+2+7+1+6+4+3}, naive_rot_first, {"first_k": 3}),
    "r52_rotid_d8_n1024":  (fold_identity, {}, naive_rot_first, {"first_k": 32}),
    "r52_rotid_d6_n1024":  (fold_identity, {}, naive_rot_first, {"first_k": 32}),
    "r52_rotid_d8_res":    (fold_identity, {}, naive_rot_first, {"first_k": 32}),
    "r52_permrot_d4_n1024": (fold_permrot, {"shifts": [3,5,2,7]}, naive_rot_first, {"first_k": 3}),
    "r52_permrot_d4_res":   (fold_permrot, {"shifts": [3,5,2,7]}, naive_rot_first, {"first_k": 3}),
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
    print(f"=== r52 collective_permute pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, (foldf, fkw, naivef, nkw) in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, foldf(name, **fkw))
            nv = score(p, naivef(name, **nkw))
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
    print("\n=== VIABLE (base+fold gate-pass, hr>=1.05) ===")
    viable = []
    for name, b_ok, b_sim, f_ok, f_sim, hr, nv_ok in rows:
        ok = bool(b_ok and f_ok and hr and hr >= 1.05)
        if ok:
            viable.append(name)
        print(f"  {name:24s} {'VIABLE' if ok else 'no    '} hr={hr} naive_fails={not nv_ok}")
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r52.json", "w"), indent=2)


if __name__ == "__main__":
    main()
