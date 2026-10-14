"""wrap_a2av_helper.py <in_runtime.py> <out_runtime.py>: make the uniform helper alltoallv(x, ws, mc) route through the
file's evolved_alltoallv (the shipped Phase-5 codegen emits a hard-coded AG+transpose+RS helper that ignores the deployed
algorithm on the OLMoE / per-primitive harness path)."""
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'^def alltoallv\(x, world_size, max_chunk\):.*?(?=^def |\Z)', src, re.S | re.M); assert m, "helper not found"
HELPER = '''def alltoallv(x, world_size, max_chunk):
    """Uniform pre-packed AllToAllV routed through the deployed evolved_alltoallv (rebuttal wrapper)."""
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")
    ws = world_size; mc = max_chunk
    return evolved_alltoallv(x, [mc] * ws, [mc] * ws, mc, _rank, ws, _NUM_DEVICES, _CORES_PER_DEVICE, xm, torch, num_nodes=_NUM_NODES)
'''
open(sys.argv[2], "w").write(src[:m.start()] + HELPER + src[m.end():]); print("wrapped ->", sys.argv[2])
