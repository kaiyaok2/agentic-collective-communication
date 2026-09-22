"""Faithful pre-screen for r55 (large-payload all_to_all transpose, fam-4 non-AR candidate).

For each problem: baseline (all_gather+slice+cat) gate-pass + sim; IDEAL fold (one all_to_all)
gate-pass + sim (headroom); and the NAIVE all_gather guess (overlay's likely one-shot: it PASSES
the gate but stays in the all_gather family, so it does NOT get the a2a headroom -- the trap is
efficiency, reachable only by switching primitive).
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


def fold_a2a(name):
    return (sig(name) +
            "    return xm.all_to_all(x, split_dimension=0, concat_dimension=0, "
            "split_count=world_size)\n")


def naive_allgather(name, S):
    # overlay's likely one-shot: correct all_gather form; passes gate but stays gather-family.
    return (sig(name) +
            f"    Sblk = {S}; W = world_size\n"
            "    g = xm.all_gather(x, dim=0)\n"
            "    parts = []\n"
            "    for k in range(W):\n"
            "        base = k * (W * Sblk)\n"
            "        parts.append(g[base + rank*Sblk : base + (rank+1)*Sblk])\n"
            "    return torch.cat(parts, dim=0)\n")


PROBS = {
    "r55_a2a_s1024": 1024,
    "r55_a2a_s2048": 2048,
    "r55_a2a_s4096": 4096,
    "r55_a2a_s8192": 8192,
    "r55_a2a_s16384": 16384,
    "r55_a2a_s32768": 32768,
    "r55_a2a_s8192_res": 8192,
    "r55_a2a_s16384_res": 16384,
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
        capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": ACC})
    return (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR:" + r.stderr[-160:])


def main():
    print(f"=== r55 large-payload all_to_all transpose pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name, S in PROBS.items():
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); continue
        try:
            b = score(p, get_baseline(name))
            f = score(p, fold_a2a(name))
            nv = score(p, naive_allgather(name, S))
        finally:
            try:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); p.stdin.flush()
            except Exception:
                pass
            p.terminate()
        b_ok, f_ok = b.get("ok"), f.get("ok")
        b_sim, f_sim = b.get("sim_time_us"), f.get("sim_time_us")
        nv_sim = nv.get("sim_time_us")
        hr = (b_sim / f_sim) if (b_ok and f_ok and f_sim) else None
        nv_hr = (nv_sim / f_sim) if (nv.get("ok") and f_sim) else None
        rows.append((name, b_ok, b_sim, f_ok, f_sim, hr, nv.get("ok"), nv_sim, nv_hr))
        note = ""
        if not b_ok:
            note = "BASE_FAIL:" + str(b.get("error"))[:90]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:90]
        print(f"{name:22s} base={str(b_sim)[:9]:>9} fold={str(f_sim)[:9]:>9} "
              f"hr={('%.3f'%hr) if hr else '-':>6} | naive_gather_ok={nv.get('ok')} "
              f"naive={str(nv_sim)[:9]:>9} a2a_vs_naive={('%.3f'%nv_hr) if nv_hr else '-':>6} {note}")
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
    print("\nVIABLE (base+fold pass, hr>=1.05):", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6], "naive_sim": r[7],
                         "a2a_vs_naive": r[8]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r55.json", "w"), indent=2)


if __name__ == "__main__":
    main()
