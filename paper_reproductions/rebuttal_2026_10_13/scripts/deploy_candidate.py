#!/usr/bin/env python3
"""Write a Phase-3 candidate into runtime/trainium_<problem>_7node.py so the per-problem
training harness (training/train_<problem>_7node.py --backend agent) measures it.

Sources:
  --builtin NAME     seed-library template from search/problems.py / search/problems_modext.py
                     (the `_<PROB>_BUILTINS["NAME"] = '''...'''` string)
  --file PATH        a complete runtime file (e.g. an archived deployed strat runtime)
The original (paper) runtime is saved once as runtime/<name>.paper.bak; --restore puts it back.
"""
import argparse, os, re, shutil, sys

REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL")
FN = {"alltoallv": "evolved_alltoallv", "uniform_a2a": "evolved_uniform_a2a",
      "ring_kv": "evolved_ring_kv", "dxe": "evolved_dxe", "pp_send_recv": "evolved_pp_send_recv",
      "tp_mlp": "evolved_tp_mlp", "fsdp_prefetch": "evolved_fsdp_prefetch",
      "llama_block_ar": "evolved_llama_block", "grad_ar": "evolved_grad_ar"}
BUILTIN_VAR = {"alltoallv": "_ALLTOALLV_BUILTINS", "uniform_a2a": "_UA2A_BUILTINS",
               "ring_kv": "_RING_KV_BUILTINS", "dxe": "_DXE_BUILTINS", "grad_ar": "_GRAD_AR_BUILTINS",
               "pp_send_recv": "_PP_BUILTINS", "tp_mlp": "_TP_MLP_BUILTINS",
               "fsdp_prefetch": "_FSDP_BUILTINS", "llama_block_ar": "_LLAMA_BLOCK_BUILTINS"}

def builtin_code(problem, name):
    pat = re.compile(r'(_[A-Z_]*BUILTINS)\["' + re.escape(name) + r'"\]\s*=\s*(\'\'\'|""")\\?\n?(.*?)\2', re.S)
    for src in ("search/problems.py", "search/problems_modext.py"):
        txt = open(os.path.join(REPO, src)).read()
        for m in pat.finditer(txt):
            code = m.group(3)
            if FN[problem] in code:
                return code
    sys.exit(f"builtin {name!r} for {problem} not found")

def replace_evolved(runtime_src, new_fn_code, fn_name):
    # cut from 'def <fn_name>(' to the next top-level 'def ' / module-level statement
    m = re.search(r'^def ' + fn_name + r'\(.*?(?=^def |^[A-Za-z_]\w* *=|\Z)', runtime_src, re.S | re.M)
    if not m:
        sys.exit(f"could not find def {fn_name} in runtime")
    return runtime_src[:m.start()] + new_fn_code.rstrip() + "\n\n\n" + runtime_src[m.end():]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=sorted(FN))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--builtin"); g.add_argument("--file"); g.add_argument("--code", help="file containing just the evolved_* function"); g.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    rt = os.path.join(REPO, "runtime", f"trainium_{a.problem}_7node.py")
    # the uniform_a2a harness imports runtime/trainium_uniform_a2a.py (NOT the _7node file); the shipped copy of that file
    # is a collective_permute ring that aborts at 224 ranks (ENC_MAX_COMM_N) -- deploy to every file the harnesses import
    targets = [rt] + ([os.path.join(REPO, "runtime", "trainium_uniform_a2a.py")] if a.problem == "uniform_a2a" else [])
    bak = rt + ".paper.bak"
    if not os.path.exists(bak):
        shutil.copy(rt, bak)
    if a.restore:
        for t in targets: shutil.copy(bak, t)
        print("restored", targets); return
    if a.file:
        for t in targets: shutil.copy(a.file, t)
        print("deployed file", a.file, "->", targets); return
    code = open(a.code).read() if a.code else builtin_code(a.problem, a.builtin)
    src = open(bak).read()
    # the public artifact's alltoallv runtime carries a stray scp 'Warning: Permanently added ...' first line (SyntaxError)
    src = "\n".join(l for l in src.splitlines() if not l.startswith("Warning: Permanently added")) + "\n"
    if a.problem == "ring_kv":
        # harness passes one (KV, HEADS, SEQ, HD) tensor; the seed templates expect a list of 1-D per-slot chunks
        code = code.replace("def evolved_ring_kv(", "def _evolved_ring_kv_impl(", 1) + (
            "\n\ndef evolved_ring_kv(kv_chunks, *args, **kwargs):\n"
            "    if hasattr(kv_chunks, 'shape'):\n"
            "        kv_chunks = [kv_chunks[i].reshape(-1) for i in range(kv_chunks.shape[0])]\n"
            "    return _evolved_ring_kv_impl(kv_chunks, *args, **kwargs)\n")
    out = replace_evolved(src, code, FN[a.problem])
    for t in targets: open(t, "w").write(out)
    print(f"deployed {'code '+a.code if a.code else 'builtin '+a.builtin} -> {targets}")

if __name__ == "__main__":
    main()
