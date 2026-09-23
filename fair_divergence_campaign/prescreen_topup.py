"""Offline pre-screen for the fam-2 (r60d) + fam-3 (r61b) top-up batteries.
No Bedrock: baseline gate-pass + ideal-fold headroom (>=1.05) + naive-guess-fails,
plus a STRICT non-duplication guard -- md5 of each new candidate at W=224 is compared
against the md5 of EVERY already-registered problem (not just within the batch), so a
collision with any confirmed r40/r43/r47/r59/r60/r61 problem is caught."""
import hashlib
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

sys.path.insert(0, ACC)
from search.problems_diverge_r60 import _emit_keep       # noqa: E402
from search.problems_diverge_r61b import _g_code as _g61b  # noqa: E402

# ---- new candidate tables ---------------------------------------------------
R60D = {
    "r60d_b9_L2_p2048_d8":  {"shape": "contig", "B": 9,  "off": 2, "L": 2, "stride": 1, "part": 2048},
    "r60d_b10_L3_p2048_d8": {"shape": "contig", "B": 10, "off": 2, "L": 3, "stride": 1, "part": 2048},
    "r60d_b10_L5_p2048_d8": {"shape": "contig", "B": 10, "off": 2, "L": 5, "stride": 1, "part": 2048},
    "r60d_b13_L3_p2048_d8": {"shape": "contig", "B": 13, "off": 2, "L": 3, "stride": 1, "part": 2048},
    "r60d_b13_L4_p2048_d8": {"shape": "contig", "B": 13, "off": 2, "L": 4, "stride": 1, "part": 2048},
    "r60d_b14_L4_p2048_d8": {"shape": "contig", "B": 14, "off": 2, "L": 4, "stride": 1, "part": 2048},
    "r60d_b16_L5_p2048_d8": {"shape": "contig", "B": 16, "off": 2, "L": 5, "stride": 1, "part": 2048},
}
R61B = {  # name -> (part, kind)
    "r61b_dd_meansq_p768_d8": (768, "meansq"),
    "r61b_dd_meansq_p1024_d8": (1024, "meansq"),
    "r61b_dd_negrelu_p512_d8": (512, "negrelu"),
    "r61b_dd_negrelu_p768_d8": (768, "negrelu"),
    "r61b_dd_negrelu_p1024_d8": (1024, "negrelu"),
    "r61b_dd_quad_p512_d8": (512, "quad"),
    "r61b_dd_quad_p768_d8": (768, "quad"),
    "r61b_dd_meansq2_p512_d8": (512, "meansq2"),
    "r61b_dd_meansq2_p768_d8": (768, "meansq2"),
    "r61b_dd_absmean_p512_d8": (512, "absmean"),
    "r61b_dd_absmean_p768_d8": (768, "absmean"),
    "r61b_dd_absmean2_p512_d8": (512, "absmean2"),
    "r61b_dd_absmean2_p768_d8": (768, "absmean2"),
}
NEW = list(R60D) + list(R61B)


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


def fold_fam2(name, spec):
    part = spec["part"]; B = spec["B"]
    L = [_sig(name).rstrip("\n"),
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         f"    c = [0]*{B}",
         "    for rank in range(W):"]
    for ln in _emit_keep(spec):
        L.append("    " + ln)
    L += ["        for b in keep:", "            c[b] += 1"]
    L += ["    out = s.clone()", f"    for b in range({B}):",
          "        out[b*S:(b+1)*S] = c[b] * s[b*S:(b+1)*S]", "    return out"]
    return "\n".join(L) + "\n"


def fold_fam3b(name, part, kind):
    L = [_sig(name).rstrip("\n"),
         f"    S = {part}; B = 8",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         _g61b(kind, indent="    "),
         "    out = s.clone()",
         "    for b in range(B):",
         "        out[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
         "    return out"]
    return "\n".join(L) + "\n"


def naive_plain(name):
    return _sig(name) + "    return xm.all_reduce(xm.REDUCE_SUM, x)\n"


def make_fold(name):
    if name in R60D:
        return fold_fam2(name, R60D[name])
    part, kind = R61B[name]
    return fold_fam3b(name, part, kind)


def all_registered():
    r = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, "
         f"{ACC!r}); import search.problems_all_catalogs;"
         "from search.problems import PROBLEMS; print('\\n'.join(sorted(PROBLEMS)))"],
        capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": ACC})
    return [x for x in r.stdout.strip().splitlines() if x]


def main():
    print(f"=== r60d/r61b top-up pre-screen @ W={WORLD} gate=fp32 ===")
    rows = []
    for name in NEW:
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
        print(f"{name:26s} base={str(b_sim)[:8]:>8} fold={str(f_sim)[:8]:>8} "
              f"hr={('%.3f' % hr) if hr else '-':>6} naive_ok={nv.get('ok')} "
              f"{'(TRAP)' if nv.get('ok') is False else '(naive PASSES!)'} {note}")

    print("\n=== STRICT non-duplication: new md5 vs ALL registered ===")
    new_h = {n: ref_hash(n) for n in NEW}
    for n, h in new_h.items():
        print(f"  {n:26s} {h[:16]}")
    # md5 of every existing registered problem (exclude the new ones)
    existing = [x for x in all_registered() if x not in R60D and x not in R61B]
    exist_h = {}
    for x in existing:
        exist_h[x] = ref_hash(x)
    # collisions: new vs existing, and new vs new
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
    # new vs new
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
              open(f"{FD}/prescreen_topup.json", "w"), indent=2)


if __name__ == "__main__":
    main()
