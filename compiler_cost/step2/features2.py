#!/usr/bin/env python3
"""features2.py — richer but still cheap structural features of an HloModuleProto for the Step 2 baselines.

On top of step0/features.py (counts, bytes, FLOPs, dtype mix) this adds: a fixed-vocabulary opcode histogram, shape-alignment
statistics (how many tensor dims are multiples of 128/32/8, odd, or >= 1024), dot / convolution shape statistics (M, N, K,
kernel and stride sizes), literal (constant) bytes, and graph-structure statistics of the entry computation (longest
dependency path, fan-out, number of roots/tuple outputs). Everything is O(#instructions).
"""
import os, sys, math, collections
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/home/ubuntu/hloproto/gen", os.path.expanduser("~/work/ncc/data/hloproto/gen")):
    if os.path.isdir(p): sys.path.insert(0, p)
sys.path.insert(0, os.path.join(HERE, "..", "step0", "scripts"))
import features as F0
from xla.service import hlo_pb2
from xla import xla_data_pb2

OPCODE_VOCAB = ["add", "multiply", "subtract", "divide", "maximum", "minimum", "exponential", "log", "tanh", "logistic", "rsqrt", "sqrt", "power", "negate", "abs",
                "select", "compare", "convert", "broadcast", "reshape", "bitcast", "transpose", "slice", "dynamic-slice", "dynamic-update-slice", "concatenate",
                "pad", "reverse", "iota", "constant", "parameter", "get-tuple-element", "tuple", "dot", "convolution", "reduce", "reduce-window", "select-and-scatter",
                "gather", "scatter", "sort", "while", "conditional", "call", "custom-call", "fusion", "rng", "rng-bit-generator", "all-reduce", "all-gather",
                "reduce-scatter", "all-to-all", "collective-permute", "copy", "clamp", "sign", "floor", "ceil", "round-nearest-afz", "and", "or", "not", "xor",
                "shift-left", "shift-right-logical", "shift-right-arithmetic", "remainder", "atan2", "cosine", "sine", "exponential-minus-one", "log-plus-one",
                "map", "reduce-precision", "optimization-barrier", "partition-id", "replica-id", "domain", "send", "recv", "infeed", "outfeed", "batch-norm-training",
                "batch-norm-inference", "batch-norm-grad", "cholesky", "triangular-solve", "fft", "real", "imag", "complex", "is-finite", "popcnt", "clz", "tan", "erf", "stochastic-convert"]

def _dims(shape, pt_name):
    return [d for _, d in F0._shape_info(shape, pt_name)]

def features2_from_pb(path):
    pt_name = lambda v: xla_data_pb2.PrimitiveType.Name(v)
    mod = hlo_pb2.HloModuleProto(); mod.ParseFromString(open(path, "rb").read())
    f = F0.features_from_pb(path)
    ops = collections.Counter(); align = collections.Counter(); n_dims = 0; literal_bytes = 0
    dot_m = []; dot_n = []; dot_k = []; conv_k = []; conv_stride = []; conv_cin = []; conv_cout = []; conv_out = []
    reduce_bytes = 0; gather_bytes = 0; transpose_bytes = 0; concat_bytes = 0; max_dim = 0; n_tuple_leaves_root = 0
    for comp in mod.computations:
        by_id = {i.id: i for i in comp.instructions}
        for ins in comp.instructions:
            ops[ins.opcode] += 1
            leaves = F0._shape_info(ins.shape, pt_name); b = F0._bytes(leaves)
            for _, d in leaves:
                for x in d:
                    n_dims += 1; max_dim = max(max_dim, x)
                    if x % 128 == 0: align["m128"] += 1
                    elif x % 32 == 0: align["m32"] += 1
                    elif x % 8 == 0: align["m8"] += 1
                    else: align["unaligned"] += 1
                    if x % 2 == 1 and x > 1: align["odd"] += 1
                    if x >= 1024: align["ge1024"] += 1
                    if x == 1: align["one"] += 1
            if ins.opcode == "constant" and ins.HasField("literal"): literal_bytes += b
            elif ins.opcode == "dot":
                try:
                    lhs = by_id[ins.operand_ids[0]]; rhs = by_id[ins.operand_ids[1]]; dn = ins.dot_dimension_numbers
                    ld = list(lhs.shape.dimensions); rd = list(rhs.shape.dimensions)
                    k = math.prod(ld[i] for i in dn.lhs_contracting_dimensions); m = math.prod(ld[i] for i in range(len(ld)) if i not in dn.lhs_contracting_dimensions and i not in dn.lhs_batch_dimensions)
                    n = math.prod(rd[i] for i in range(len(rd)) if i not in dn.rhs_contracting_dimensions and i not in dn.rhs_batch_dimensions)
                    dot_m.append(m); dot_n.append(n); dot_k.append(k)
                except Exception: pass
            elif ins.opcode == "convolution":
                try:
                    rhs = by_id[ins.operand_ids[1]]; dn = ins.convolution_dimension_numbers; kd = list(rhs.shape.dimensions)
                    conv_k.append(math.prod(kd[i] for i in dn.kernel_spatial_dimensions)); conv_cin.append(kd[dn.kernel_input_feature_dimension]); conv_cout.append(kd[dn.kernel_output_feature_dimension])
                    conv_stride.append(max([w.stride for w in ins.window.dimensions] or [1])); conv_out.append(math.prod(leaves[0][1]) if leaves and leaves[0][1] else 1)
                except Exception: pass
            elif ins.opcode in ("reduce", "reduce-window"): reduce_bytes += sum(F0._bytes(F0._shape_info(by_id[o].shape, pt_name)) for o in ins.operand_ids[:1] if o in by_id)
            elif ins.opcode in ("gather", "scatter", "dynamic-slice", "dynamic-update-slice"): gather_bytes += b
            elif ins.opcode == "transpose": transpose_bytes += b
            elif ins.opcode == "concatenate": concat_bytes += b
    # graph structure of the entry computation
    entry = {c.id: c for c in mod.computations}.get(mod.entry_computation_id)
    depth = 0; fanout = collections.Counter(); n_roots = 0
    if entry is not None:
        by_id = {i.id: i for i in entry.instructions}; dep = {}
        for ins in entry.instructions:   # instructions are topologically ordered in the proto
            d = 0
            for o in ins.operand_ids:
                fanout[o] += 1
                if o in dep: d = max(d, dep[o] + 1)
            dep[ins.id] = d
        depth = max(dep.values()) if dep else 0
        used = set(fanout); n_roots = sum(1 for i in entry.instructions if i.id not in used)
        root = by_id.get(entry.root_id); n_tuple_leaves_root = len(F0._shape_info(root.shape, pt_name)) if root is not None else 0
    fo = [fanout[i.id] for i in (entry.instructions if entry is not None else [])]
    def stats(v, prefix):
        if not v: return {f"{prefix}_max": 0, f"{prefix}_sum": 0, f"{prefix}_mean": 0, f"{prefix}_n": 0}
        return {f"{prefix}_max": max(v), f"{prefix}_sum": sum(v), f"{prefix}_mean": sum(v) / len(v), f"{prefix}_n": len(v)}
    f2 = collections.OrderedDict()
    f2.update({f"op_{o}": ops.get(o, 0) for o in OPCODE_VOCAB}); f2["op_other"] = sum(v for k, v in ops.items() if k not in OPCODE_VOCAB)
    f2.update(dict(n_dims=n_dims, max_dim=max_dim, literal_bytes=literal_bytes, reduce_bytes=reduce_bytes, gather_bytes=gather_bytes, transpose_bytes=transpose_bytes, concat_bytes=concat_bytes,
                   depth=depth, n_roots=n_roots, root_leaves=n_tuple_leaves_root, fanout_max=max(fo) if fo else 0, fanout_mean=(sum(fo) / len(fo)) if fo else 0,
                   frac_dims_m128=align["m128"] / max(1, n_dims), frac_dims_m32=align["m32"] / max(1, n_dims), frac_dims_m8=align["m8"] / max(1, n_dims), frac_dims_unaligned=align["unaligned"] / max(1, n_dims),
                   n_dims_odd=align["odd"], n_dims_ge1024=align["ge1024"], n_dims_one=align["one"]))
    f2.update(stats(dot_m, "dot_m")); f2.update(stats(dot_n, "dot_n")); f2.update(stats(dot_k, "dot_k")); f2.update(stats(conv_k, "conv_kernel")); f2.update(stats(conv_cin, "conv_cin")); f2.update(stats(conv_cout, "conv_cout")); f2.update(stats(conv_out, "conv_out")); f2["conv_stride_max"] = max(conv_stride) if conv_stride else 0
    f.update(f2)
    return f

if __name__ == "__main__":
    import json
    for p in sys.argv[1:]: print(json.dumps(features2_from_pb(p), indent=0)[:3000])
