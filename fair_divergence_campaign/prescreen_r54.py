"""Faithful pre-screen for r54 (per-block max-plus telescoping w/ awkward unshift)."""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32
NBLOCK = 8


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


def fold(name, part, depth, gamma, is_max):
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    ext = "max" if is_max else "min"
    return (sig(name) +
            f"    S = {part}; B = {NBLOCK}; W = world_size; G = {gamma}; D = {depth}\n"
            "    beta = torch.zeros(B*S)\n"
            "    for b in range(B):\n"
            "        val = G * ((((rank * 37) + b * 13) % W) - W/2.0) / W\n"
            "        beta[b*S:(b+1)*S] = val\n"
            f"    m = xm.all_reduce({red}, x + beta)\n"
            "    net = torch.zeros(B*S)\n"
            "    for b in range(B):\n"
            "        col = [G*((((rr*37)+b*13) % W) - W/2.0)/W for rr in range(W)]\n"
            f"        colext = {ext}(col)\n"
            "        ub = 0.1 + 0.07 * ((b*5) % 6)\n"
            "        net[b*S:(b+1)*S] = (D-1) * (colext - ub)\n"
            "    return m + net\n")


def naive(name, is_max):
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    return sig(name) + f"    return xm.all_reduce({red}, x)\n"


PROBS = {
    "r54_bmax_d8_p256": (256, 8, 2.0, True),
    "r54_bmax_d6_p256": (256, 6, 2.0, True),
    "r54_bmax_d4_p256": (256, 4, 2.0, True),
    "r54_bmax_d8_g3": (256, 8, 3.0, True),
    "r54_bmax_d8_g5": (256, 8, 5.0, True),
    "r54_bmax_d8_p512": (512, 8, 2.0, True),
    "r54_bmin_d8_p256": (256, 8, 2.0, False),
    "r54_bmax_d8_res": (256, 8, 2.0, True),
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
    print(f"=== r54 per-block max-plus pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, (part, depth, gamma, is_max) in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, fold(name, part, depth, gamma, is_max))
            nv = score(p, naive(name, is_max))
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
              f"{'(TRAP)' if not nv.get('ok') else '(guess PASSES)'} {note}")
    print("\n=== distinctness ===")
    hashes = {}
    for name, *_ in rows:
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
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r54.json", "w"), indent=2)


if __name__ == "__main__":
    main()
