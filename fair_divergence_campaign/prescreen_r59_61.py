"""Faithful offline pre-screen for the family-expansion batteries r59 (fam-1),
r60 (fam-2), r61 (fam-3). No Bedrock: only baseline gate-pass + ideal-fold headroom
(>=1.05) + naive-guess-fails + md5 distinctness at W=224. Mirrors prescreen_r58."""
import json, os, subprocess, sys

FD = "/private/tmp/fair_diverge"
ACC = "/private/tmp/acc_verify"
PY = sys.executable
SCORE_PY = f"{FD}/score_service_fair.py"
NODES = 7
WORLD = NODES * 32

sys.path.insert(0, ACC)
from search.problems_diverge_r60 import _emit_keep  # noqa: E402


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


def _sig(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            f"                 cores_per_device, xm, torch, num_nodes=1):\n")


# ---- fold emitters (ideal 1-collective form) --------------------------------
def fold_fam1(name, part, a_expr):
    return (_sig(name) +
            f"    S = {part}; W = world_size\n"
            f"    a = [{a_expr} for r in range(W)]\n"
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)\n"
            "    out = s.clone()\n"
            "    for r in range(W):\n"
            "        out[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S]\n"
            "    return out\n")


def fold_fam2(name, spec):
    part = spec["part"]; B = spec["B"]
    L = [_sig(name).rstrip("\n"),
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         f"    c = [0]*{B}",
         "    for rank in range(W):"]
    # reuse the exact keep-set emitter (indented into the r-loop), aliasing rank
    for ln in _emit_keep(spec):
        L.append("    " + ln)
    L += ["        for b in keep:",
          "            c[b] += 1"]
    L += ["    out = s.clone()",
          f"    for b in range({B}):",
          "        out[b*S:(b+1)*S] = c[b] * s[b*S:(b+1)*S]",
          "    return out"]
    return "\n".join(L) + "\n"


def fold_fam3_dd(name, part, kind):
    from search.problems_diverge_r61 import _g_code
    L = [_sig(name).rstrip("\n"),
         f"    S = {part}; B = 8",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         _g_code(kind, indent="    "),
         "    out = s.clone()",
         "    for b in range(B):",
         "        out[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
         "    return out"]
    return "\n".join(L) + "\n"


def fold_fam3_xc(name, rounds):
    return (_sig(name) +
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)\n"
            f"    return s * (world_size ** {rounds - 1})\n")


def naive_plain(name):
    return _sig(name) + "    return xm.all_reduce(xm.REDUCE_SUM, x)\n"


# ---- problem table: name -> ("fam1"/"fam2"/"fam3dd"/"fam3xc", params) --------
R59 = {
    "r59_su_a5_d8": ("fam1", 256, "1.0 + 0.4*(r % 5)"),
    "r59_su_a4_d6_p512": ("fam1", 512, "1.0 + 0.6*(r % 4)"),
    "r59_su_a7_d8": ("fam1", 256, "1.0 + 0.3*(r % 7)"),
    "r59_su_a4_d8_p1024": ("fam1", 1024, "1.0 + 0.6*(r % 4)"),
}
R60 = {
    "r60_b5_L2_d8": {"shape": "contig", "B": 5, "off": 0, "L": 2, "stride": 1, "part": 256},
    "r60_b5_L3_d8": {"shape": "contig", "B": 5, "off": 0, "L": 3, "stride": 1, "part": 256},
    "r60_b5_o1_L2_d8": {"shape": "contig", "B": 5, "off": 1, "L": 2, "stride": 1, "part": 256},
    "r60_b6_L3_d8": {"shape": "contig", "B": 6, "off": 0, "L": 3, "stride": 1, "part": 256},
    "r60_b6_L4_d8": {"shape": "contig", "B": 6, "off": 0, "L": 4, "stride": 1, "part": 256},
    "r60_b6_o2_L3_d8": {"shape": "contig", "B": 6, "off": 2, "L": 3, "stride": 1, "part": 256},
    "r60_b6_L5_d8": {"shape": "contig", "B": 6, "off": 0, "L": 5, "stride": 1, "part": 256},
    "r60_b6_varL_d8": {"shape": "varL", "B": 6, "off": 0, "lmin": 3, "lspan": 2, "L": 3, "stride": 1, "part": 256},
    "r60_b5_varL_d8": {"shape": "varL", "B": 5, "off": 0, "lmin": 2, "lspan": 2, "L": 2, "stride": 1, "part": 256},
}
R61 = {
    "r61_dd_var_d8": ("fam3dd", 256, "var"),
    "r61_dd_soft_d8": ("fam3dd", 256, "soft"),
    "r61_dd_negrelu_d8": ("fam3dd", 256, "negrelu"),
    "r61_dd_var_p384_d8": ("fam3dd", 384, "var"),
    "r61_dd_soft_p512_d8": ("fam3dd", 512, "soft"),
    "r61_dd_negrelu_p384_d8": ("fam3dd", 384, "negrelu"),
    "r61_dd_meansq_p512_d8": ("fam3dd", 512, "meansq"),
    "r61_dd_relu_p384_d8": ("fam3dd", 384, "relu"),
}


def make_fold(name):
    if name in R59:
        _, part, a = R59[name]; return fold_fam1(name, part, a)
    if name in R60:
        return fold_fam2(name, R60[name])
    fam = R61[name][0]
    if fam == "fam3dd":
        _, part, kind = R61[name]; return fold_fam3_dd(name, part, kind)
    _, part, rounds = R61[name]; return fold_fam3_xc(name, rounds)


def main():
    print(f"=== r59/r60/r61 family-expansion pre-screen @ W={WORLD} gate=fp32 ===")
    names = list(R59) + list(R60) + list(R61)
    rows = []
    for name in names:
        try:
            p = start_scorer(name)
        except Exception as e:
            print(f"{name}: SCORER FAIL {str(e)[:150]}"); rows.append((name, False, None, False, None, None, None)); continue
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
            note = "BASE_FAIL:" + str(b.get("error"))[:110]
        elif not f_ok:
            note = "FOLD_FAIL:" + str(f.get("error"))[:110]
        print(f"{name:24s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f'%hr) if hr else '-':>6} naive_ok={nv.get('ok')} "
              f"{'(TRAP)' if not nv.get('ok') else '(naive PASSES!)'} {note}")
    print("\n=== distinctness (md5 @ seed 12345) ===")
    hashes = {}
    for r in rows:
        h = ref_hash(r[0]); hashes[r[0]] = h
        print(f"  {r[0]:24s} {h[:16]}")
    inv = {}
    for n, h in hashes.items():
        inv.setdefault(h, []).append(n)
    dups = {h: ns for h, ns in inv.items() if len(ns) > 1 and not h.startswith("ERR")}
    print("  collisions:", {h[:8]: ns for h, ns in dups.items()} if dups else "NONE")
    viable = [r[0] for r in rows if (r[1] and r[3] and r[6] is False and r[5] and r[5] >= 1.05)]
    print("\nVIABLE:", ",".join(viable))
    json.dump({"rows": [{"prob": r[0], "base_ok": r[1], "base_sim": r[2], "fold_ok": r[3],
                         "fold_sim": r[4], "headroom": r[5], "naive_ok": r[6]} for r in rows],
               "hashes": hashes, "viable": viable}, open(f"{FD}/prescreen_r59_61.json", "w"), indent=2)


if __name__ == "__main__":
    main()
