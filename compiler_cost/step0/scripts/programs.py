"""programs.py — registry of graph-producing programs for the compiler-cost corpus.

A *spec* is a JSON-able dict: {"id": str, "family": str, "source": "synth|real|mutation|collective", "params": {...},
"parent": optional id, "mutation": optional str}. `build(spec, dev)` returns (step_fn, keep) where step_fn() runs one
iteration on device `dev` and returns a list of output tensors; `keep` is a list of tensors (parameters) that stay live.
Everything is random-initialised; no downloads.
"""
import math, random, itertools, os, torch

# ----------------------------------------------------------------------------- helpers
def _dtype(name): return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]

class MLP(torch.nn.Module):
    def __init__(s, d, hidden, depth, act):
        super().__init__(); layers = []; last = d
        for i in range(depth):
            layers += [torch.nn.Linear(last, hidden), {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "tanh": torch.nn.Tanh(), "none": torch.nn.Identity()}[act]]; last = hidden
        layers += [torch.nn.Linear(last, d)]; s.net = torch.nn.Sequential(*layers)
    def forward(s, x): return s.net(x)

class Attn(torch.nn.Module):
    def __init__(s, dim, heads, layers, ffn_mult=4, norm=True, causal=True):
        super().__init__(); s.h = heads; s.causal = causal
        s.blocks = torch.nn.ModuleList([torch.nn.ModuleDict(dict(qkv=torch.nn.Linear(dim, 3 * dim), o=torch.nn.Linear(dim, dim),
                                                                ff1=torch.nn.Linear(dim, ffn_mult * dim), ff2=torch.nn.Linear(ffn_mult * dim, dim),
                                                                n1=torch.nn.LayerNorm(dim) if norm else torch.nn.Identity(), n2=torch.nn.LayerNorm(dim) if norm else torch.nn.Identity())) for _ in range(layers)])
    def forward(s, x):
        B, T, D = x.shape
        for b in s.blocks:
            h = b["n1"](x); q, k, v = b["qkv"](h).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // s.h)
            if s.causal: att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), float("-inf"))
            y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, D); x = x + b["o"](y)
            x = x + b["ff2"](torch.nn.functional.gelu(b["ff1"](b["n2"](x))))
        return x

class ConvStack(torch.nn.Module):
    def __init__(s, ch, layers, in_ch=3, bn=True):
        super().__init__(); mods = []; last = in_ch
        for i in range(layers):
            mods += [torch.nn.Conv2d(last, ch, 3, padding=1), torch.nn.BatchNorm2d(ch) if bn else torch.nn.Identity(), torch.nn.ReLU()]; last = ch
            if i % 2 == 1: mods += [torch.nn.MaxPool2d(2)]
        s.net = torch.nn.Sequential(*mods); s.head = torch.nn.Linear(ch, 10)
    def forward(s, x): return s.head(s.net(x).mean((2, 3)))

def _train_step(model, x, target_fn, lr=1e-3):
    params = [p for p in model.parameters()]
    def step():
        out = model(x); loss = target_fn(out)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None: p.sub_(lr * g)
        return [loss]
    return step, params

# ----------------------------------------------------------------------------- builders per family
def b_synth_mlp(p, dev):
    dt = _dtype(p.get("dtype", "fp32")); m = MLP(p["d"], p["hidden"], p["depth"], p.get("act", "relu")).to(dev, dt)
    x = torch.randn(p["batch"], p["d"], device=dev, dtype=dt)
    if p.get("train"): return _train_step(m, x, lambda o: (o.float() ** 2).mean())
    return (lambda: [m(x)]), list(m.parameters())

def b_synth_attn(p, dev):
    dt = _dtype(p.get("dtype", "fp32")); m = Attn(p["dim"], p["heads"], p["layers"], norm=p.get("norm", True), causal=p.get("causal", True)).to(dev, dt)
    x = torch.randn(p["batch"], p["seq"], p["dim"], device=dev, dtype=dt)
    if p.get("train"): return _train_step(m, x, lambda o: (o.float() ** 2).mean())
    return (lambda: [m(x)]), list(m.parameters())

def b_synth_elementwise(p, dev):
    rng = random.Random(p["seed"]); n = p["numel"]; x = torch.randn(n, device=dev); y = torch.randn(n, device=dev)
    ops = ["add", "mul", "sin", "exp", "tanh", "where", "clamp", "sub", "div", "maximum", "sigmoid", "abs"]
    seq = [rng.choice(ops) for _ in range(p["n_ops"])]
    def step():
        a = x
        for o in seq:
            if o == "add": a = a + y
            elif o == "mul": a = a * y
            elif o == "sin": a = torch.sin(a)
            elif o == "exp": a = torch.exp(torch.clamp(a, -5, 5))
            elif o == "tanh": a = torch.tanh(a)
            elif o == "where": a = torch.where(a > 0, a, y)
            elif o == "clamp": a = torch.clamp(a, -1, 1)
            elif o == "sub": a = a - y
            elif o == "div": a = a / (y.abs() + 1)
            elif o == "maximum": a = torch.maximum(a, y)
            elif o == "sigmoid": a = torch.sigmoid(a)
            elif o == "abs": a = torch.abs(a)
        return [a]
    return step, []

def b_synth_reduce(p, dev):
    x = torch.randn(*p["shape"], device=dev)
    kinds = p["kinds"]
    def step():
        outs = []
        for k in kinds:
            if k == "sum0": outs.append(x.sum(0))
            elif k == "sum_last": outs.append(x.sum(-1))
            elif k == "mean_all": outs.append(x.mean())
            elif k == "max1": outs.append(x.max(1).values)
            elif k == "softmax_last": outs.append(torch.softmax(x, -1))
            elif k == "logsumexp0": outs.append(torch.logsumexp(x, 0))
            elif k == "var_last": outs.append(x.var(-1))
            elif k == "cumsum_last": outs.append(torch.cumsum(x, -1))
            elif k == "norm": outs.append(torch.linalg.vector_norm(x, dim=-1))
            elif k == "argmax_last": outs.append(x.argmax(-1))
        return outs
    return step, []

def b_synth_conv(p, dev):
    m = ConvStack(p["ch"], p["layers"], bn=p.get("bn", True)).to(dev); x = torch.randn(p["batch"], 3, p["img"], p["img"], device=dev)
    if p.get("train"): return _train_step(m, x, lambda o: (o ** 2).mean())
    return (lambda: [m(x)]), list(m.parameters())

def b_synth_gather(p, dev):
    V, D, N = p["vocab"], p["dim"], p["n_idx"]; table = torch.randn(V, D, device=dev); idx = torch.randint(0, V, (N,), device=dev)
    src = torch.randn(N, D, device=dev); kind = p["kind"]
    def step():
        if kind == "index_select": return [table.index_select(0, idx)]
        if kind == "embedding": return [torch.nn.functional.embedding(idx, table)]
        if kind == "gather": return [torch.gather(table, 0, idx.unsqueeze(1).expand(N, D))]
        if kind == "index_add": return [torch.zeros(V, D, device=dev).index_add(0, idx, src)]
        if kind == "scatter_add": return [torch.zeros(V, D, device=dev).scatter_add(0, idx.unsqueeze(1).expand(N, D), src)]
        if kind == "one_hot_matmul": return [torch.nn.functional.one_hot(idx, V).to(table.dtype) @ table]
        if kind == "take_along": return [torch.take_along_dim(table, idx.unsqueeze(1).expand(N, D), 0)]
        raise ValueError(kind)
    return step, []

def b_synth_sort(p, dev):
    x = torch.randn(*p["shape"], device=dev); kind = p["kind"]; k = p.get("k", 8)
    def step():
        if kind == "sort": return list(torch.sort(x, -1))
        if kind == "topk": return list(torch.topk(x, k, -1))
        if kind == "argsort": return [torch.argsort(x, -1)]
        if kind == "cumsum": return [torch.cumsum(x, -1)]
        if kind == "unique_like": return [torch.sort(x.flatten())[0][::7]]
        raise ValueError(kind)
    return step, []

def b_synth_many_small(p, dev):
    n, m = p["n_mats"], p["mat"]; dt = _dtype(p.get("dtype", "fp32"))
    xs = [torch.randn(m, m, device=dev, dtype=dt) for _ in range(n)]; w = torch.randn(m, m, device=dev, dtype=dt)
    mode = p["mode"]
    def step():
        if mode == "unrolled": return [x @ w for x in xs]
        if mode == "unrolled_chain":
            a = xs[0]
            for x in xs[1:]: a = (a @ w) + x
            return [a]
        if mode == "batched": return [torch.stack(xs) @ w]
        if mode == "distinct_shapes": return [xs[i][: m - i % 7, :] @ w for i in range(n)]
        raise ValueError(mode)
    return step, []

def b_synth_reshape(p, dev):
    x = torch.randn(*p["shape"], device=dev); n = p["n_ops"]; rng = random.Random(p["seed"])
    def step():
        a = x
        for i in range(n):
            op = rng.choice(["t", "r", "c", "s", "f"])
            if op == "t": a = a.transpose(0, -1).contiguous()
            elif op == "r": a = a.reshape(-1, a.shape[-1]).reshape(a.shape)
            elif op == "c": a = torch.cat([a, a], 0)[: a.shape[0]]
            elif op == "s": a = torch.roll(a, 1, 0)
            elif op == "f": a = a.flip(-1)
        return [a]
    return step, []

def b_synth_random_dag(p, dev):
    """seeded random DAG over 2-D tensors of one shape: matmul/add/mul/act/reduce-broadcast/transposes"""
    rng = random.Random(p["seed"]); n = p["n_nodes"]; m = p["m"]; w = p.get("width", 3)
    inputs = [torch.randn(m, m, device=dev) for _ in range(w)]
    plan = []
    for i in range(n):
        k = rng.choice(["mm", "add", "mul", "gelu", "ln", "softmax", "t", "rb"]); a = rng.randrange(w + i); b = rng.randrange(w + i); plan.append((k, a, b))
    def step():
        vals = list(inputs)
        for k, a, b in plan:
            x, y = vals[a], vals[b]
            if k == "mm": vals.append(x @ y)
            elif k == "add": vals.append(x + y)
            elif k == "mul": vals.append(x * y)
            elif k == "gelu": vals.append(torch.nn.functional.gelu(x))
            elif k == "ln": vals.append(torch.nn.functional.layer_norm(x, (m,)))
            elif k == "softmax": vals.append(torch.softmax(x, -1))
            elif k == "t": vals.append(x.t().contiguous())
            elif k == "rb": vals.append(x - x.mean(-1, keepdim=True))
        return [vals[-1]] + [vals[-i] for i in range(2, min(4, len(vals)))]
    return step, []

def b_hf(p, dev):
    import transformers
    arch = p["arch"]; dt = _dtype(p.get("dtype", "fp32")); B, T = p["batch"], p.get("seq", 64)
    if arch == "gpt2":
        cfg = transformers.GPT2Config(n_layer=p["layers"], n_embd=p["hidden"], n_head=p["heads"], n_positions=max(T, 64), vocab_size=p.get("vocab", 50257), use_cache=False)
        m = transformers.GPT2LMHeadModel(cfg); lm = True
    elif arch == "llama":
        cfg = transformers.LlamaConfig(num_hidden_layers=p["layers"], hidden_size=p["hidden"], num_attention_heads=p["heads"], num_key_value_heads=p.get("kv_heads", p["heads"]),
                                       intermediate_size=p.get("inter", 4 * p["hidden"]), vocab_size=p.get("vocab", 32000), max_position_embeddings=max(T, 64), use_cache=False)
        m = transformers.LlamaForCausalLM(cfg); lm = True
    elif arch == "bert":
        cfg = transformers.BertConfig(num_hidden_layers=p["layers"], hidden_size=p["hidden"], num_attention_heads=p["heads"], intermediate_size=4 * p["hidden"], vocab_size=p.get("vocab", 30522), max_position_embeddings=max(T, 64))
        m = transformers.BertForMaskedLM(cfg); lm = True
    elif arch == "t5":
        cfg = transformers.T5Config(num_layers=p["layers"], num_decoder_layers=p["layers"], d_model=p["hidden"], num_heads=p["heads"], d_ff=4 * p["hidden"], d_kv=p["hidden"] // p["heads"], vocab_size=p.get("vocab", 32128), use_cache=False)
        cfg.decoder_start_token_id = 0; m = transformers.T5ForConditionalGeneration(cfg); lm = "t5"
    elif arch == "vit":
        cfg = transformers.ViTConfig(num_hidden_layers=p["layers"], hidden_size=p["hidden"], num_attention_heads=p["heads"], intermediate_size=4 * p["hidden"], image_size=p.get("img", 224), patch_size=16, num_labels=1000)
        m = transformers.ViTForImageClassification(cfg); lm = False
    elif arch == "mixtral":
        cfg = transformers.MixtralConfig(num_hidden_layers=p["layers"], hidden_size=p["hidden"], num_attention_heads=p["heads"], num_key_value_heads=p.get("kv_heads", p["heads"]),
                                         intermediate_size=p.get("inter", 2 * p["hidden"]), vocab_size=p.get("vocab", 32000), num_local_experts=p.get("experts", 8), num_experts_per_tok=p.get("topk", 2), max_position_embeddings=max(T, 64), use_cache=False)
        m = transformers.MixtralForCausalLM(cfg); lm = True
    else: raise ValueError(arch)
    m = m.to(dev, dt); m.train(bool(p.get("train")))
    if arch == "vit":
        x = torch.randn(B, 3, p.get("img", 224), p.get("img", 224), device=dev, dtype=dt); labels = torch.randint(0, 1000, (B,), device=dev)
        fwd = lambda: m(pixel_values=x, labels=labels if p.get("train") else None)
    elif lm == "t5":
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=dev); dec = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        fwd = lambda: m(input_ids=ids, labels=dec) if p.get("train") else m(input_ids=ids, decoder_input_ids=dec)
    else:
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        fwd = lambda: m(input_ids=ids, labels=ids) if p.get("train") else m(input_ids=ids)
    params = [q for q in m.parameters()]
    if p.get("train"):
        def step():
            out = fwd(); loss = out.loss.float(); grads = torch.autograd.grad(loss, params, allow_unused=True)
            with torch.no_grad():
                for q, g in zip(params, grads):
                    if g is not None: q.sub_(1e-3 * g)
            return [loss]
        return step, params
    return (lambda: [fwd().logits]), params

def b_collective(p, dev):
    """ws>1 programs (run under torchrun): all_reduce / all_gather / reduce_scatter chains"""
    import torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    ws = xr.world_size(); n = p["numel"]; k = p["kind"]; cnt = p.get("count", 1)
    xs = [torch.randn(n, device=dev) for _ in range(cnt)]
    def step():
        outs = []
        for x in xs:
            if k == "all_reduce": outs.append(xm.all_reduce("sum", x))
            elif k == "all_gather": outs.append(xm.all_gather(x))
            elif k == "reduce_scatter": outs.append(xm.reduce_scatter("sum", x, 1.0, 0, ws))
            elif k == "ag_rs": outs.append(xm.reduce_scatter("sum", xm.all_gather(x), 1.0, 0, ws))
            elif k == "stacked_ar": return [xm.all_reduce("sum", torch.stack(xs))]
        return outs
    return step, []

BUILDERS = dict(synth_mlp=b_synth_mlp, synth_attn=b_synth_attn, synth_elementwise=b_synth_elementwise, synth_reduce=b_synth_reduce, synth_conv=b_synth_conv,
                synth_gather=b_synth_gather, synth_sort=b_synth_sort, synth_many_small=b_synth_many_small, synth_reshape=b_synth_reshape, synth_random_dag=b_synth_random_dag,
                hf=b_hf, collective=b_collective)

def build(spec, dev): return BUILDERS[spec["family"]](spec["params"], dev)

# ----------------------------------------------------------------------------- spec grids
def _spec(family, source, **params):
    tag = "_".join(f"{k}{v}" if not isinstance(v, (list, tuple)) else f"{k}{'x'.join(map(str, v))}" for k, v in params.items())
    tag = tag.replace("True", "1").replace("False", "0").replace(".", "p")
    return {"id": f"{family}__{tag}", "family": family, "source": source, "params": params}

def grid_synth():
    S = []
    for depth, hidden, batch, train in itertools.product([1, 4, 16], [256, 2048], [64, 1024], [False, True]):
        S.append(_spec("synth_mlp", "synth", d=512, hidden=hidden, depth=depth, batch=batch, act="gelu", train=train))
    for seq, dim, layers, train in itertools.product([128, 1024], [512, 1024], [1, 4], [False, True]):
        S.append(_spec("synth_attn", "synth", dim=dim, heads=8, layers=layers, seq=seq, batch=4, train=train))
    for n_ops, numel, seed in itertools.product([8, 64, 512], [100_000, 10_000_000], [0, 1]):
        S.append(_spec("synth_elementwise", "synth", n_ops=n_ops, numel=numel, seed=seed))
    for shape in ([4096, 4096], [64, 1024, 256], [1_000_000]):
        for kinds in (["sum0", "sum_last", "mean_all"], ["softmax_last", "logsumexp0", "var_last"], ["cumsum_last", "argmax_last", "norm", "max1"]):
            if len(shape) == 1 and any(k in ("sum0", "max1", "logsumexp0") for k in kinds): continue
            S.append(_spec("synth_reduce", "synth", shape=shape, kinds=kinds))
    for ch, layers, img, train in itertools.product([32, 128], [2, 8], [64], [False, True]):
        S.append(_spec("synth_conv", "synth", ch=ch, layers=layers, img=img, batch=8, train=train))
    for kind, vocab, n_idx in itertools.product(["index_select", "embedding", "gather", "index_add", "scatter_add", "one_hot_matmul"], [1024, 32768], [256, 8192]):
        if kind == "one_hot_matmul" and vocab == 32768 and n_idx == 8192: continue
        S.append(_spec("synth_gather", "synth", kind=kind, vocab=vocab, dim=256, n_idx=n_idx))
    for kind, shape in itertools.product(["sort", "topk", "argsort", "cumsum"], ([1024, 1024], [64, 65536], [8, 8, 4096])):
        S.append(_spec("synth_sort", "synth", kind=kind, shape=shape, k=8))
    for n_mats, mat, mode in itertools.product([16, 128, 1024], [64, 256], ["unrolled", "unrolled_chain", "batched", "distinct_shapes"]):
        if n_mats == 1024 and mat == 256 and mode != "batched": continue
        S.append(_spec("synth_many_small", "synth", n_mats=n_mats, mat=mat, mode=mode))
    for shape, n_ops, seed in itertools.product(([1024, 1024], [16, 64, 1024]), [8, 64], [0, 1]):
        S.append(_spec("synth_reshape", "synth", shape=shape, n_ops=n_ops, seed=seed))
    for n_nodes, m, seed in itertools.product([20, 100, 400], [256, 1024], [0, 1, 2]):
        if n_nodes == 400 and m == 1024 and seed > 0: continue
        S.append(_spec("synth_random_dag", "synth", n_nodes=n_nodes, m=m, seed=seed))
    return S

def grid_real():
    S = []
    for layers, train in itertools.product([1, 2, 6, 12], [False, True]):
        S.append(_spec("hf", "real", arch="gpt2", layers=layers, hidden=768, heads=12, batch=2, seq=512, train=train))
    for layers, hidden, train in itertools.product([1, 4], [512, 2048], [False, True]):
        S.append(_spec("hf", "real", arch="llama", layers=layers, hidden=hidden, heads=hidden // 64, batch=1, seq=512, train=train))
    for layers, train in itertools.product([2, 12], [False, True]):
        S.append(_spec("hf", "real", arch="bert", layers=layers, hidden=768, heads=12, batch=4, seq=128, train=train))
    for layers, train in itertools.product([2, 6], [False, True]):
        S.append(_spec("hf", "real", arch="t5", layers=layers, hidden=512, heads=8, batch=2, seq=128, train=train))
    for layers, train in itertools.product([2, 12], [False, True]):
        S.append(_spec("hf", "real", arch="vit", layers=layers, hidden=768, heads=12, batch=4, img=224, train=train))
    for layers, experts, train in itertools.product([1, 2], [4, 16], [False, True]):
        S.append(_spec("hf", "real", arch="mixtral", layers=layers, hidden=512, heads=8, kv_heads=2, inter=1024, experts=experts, topk=2, batch=1, seq=256, train=train))
    return S

def grid_mutations(bases=None):
    """near-identical pairs: one parameter perturbed from a base spec"""
    S = []
    bases = bases or [
        _spec("synth_mlp", "synth", d=512, hidden=2048, depth=4, batch=1024, act="gelu", train=True),
        _spec("synth_attn", "synth", dim=1024, heads=8, layers=4, seq=1024, batch=4, train=True),
        _spec("hf", "real", arch="gpt2", layers=2, hidden=768, heads=12, batch=2, seq=512, train=True),
        _spec("hf", "real", arch="llama", layers=4, hidden=512, heads=8, batch=1, seq=512, train=True),
        _spec("synth_conv", "synth", ch=128, layers=8, img=64, batch=8, train=True),
    ]
    for b in bases:
        p = b["params"]
        muts = []
        for key in ("hidden", "dim", "seq", "batch", "d"):
            if key in p:
                for delta_name, f in (("plus1", lambda v: v + 1), ("minus1", lambda v: v - 1), ("x2", lambda v: v * 2), ("half", lambda v: max(8, v // 2))):
                    muts.append((f"{key}_{delta_name}", {key: f(p[key])}))
        for key in ("depth", "layers"):
            if key in p: muts.append((f"{key}_plus1", {key: p[key] + 1}))
        if "act" in p: muts.append(("act_relu", {"act": "relu"})); muts.append(("act_none", {"act": "none"}))
        if p.get("dtype", "fp32") == "fp32": muts.append(("dtype_bf16", {"dtype": "bf16"}))
        if "train" in p: muts.append(("train_off", {"train": False}))
        if "norm" in p or b["family"] == "synth_attn": muts.append(("norm_off", {"norm": False})); muts.append(("causal_off", {"causal": False}))
        for name, change in muts:
            q = dict(p); q.update(change)
            if b["family"] == "hf" and "hidden" in change: q["heads"] = max(1, q["hidden"] // 64) if q["hidden"] % 64 == 0 else q["heads"]
            if b["family"] == "hf" and "hidden" in change and q["hidden"] % q["heads"] != 0: continue
            if b["family"] == "synth_attn" and q["dim"] % q["heads"] != 0: continue
            s = _spec(b["family"], "mutation", **q); s["id"] = f"mut__{b['id']}__{name}"; s["parent"] = b["id"]; s["mutation"] = name; S.append(s)
    return S

def grid_collective():
    S = []
    for kind, numel, count in itertools.product(["all_reduce", "all_gather", "reduce_scatter", "ag_rs", "stacked_ar"], [4096, 1_048_576, 16_777_216], [1, 8]):
        if kind == "stacked_ar" and count == 1: continue
        S.append(_spec("collective", "collective", kind=kind, numel=numel, count=count))
    return S

GRIDS = dict(synth=grid_synth, real=grid_real, mutation=grid_mutations, collective=grid_collective)

if __name__ == "__main__":
    import json, sys
    for g in (sys.argv[1:] or GRIDS):
        specs = GRIDS[g](); print(g, len(specs))
        for s in specs[:3]: print("  ", json.dumps(s))


# ============================================================================= Step 1 additions (2026-10-23)
# 1. systematic one-dimensional sweeps around tiling / power-of-two boundaries (near-identical graph series)
# 2. new synthetic operator families (normalization, layout, dynamic slicing, masking, scatter, conv variants, einsum,
#    unrolled recurrence, large-vocab logits, dtype casts)
# 3. collective-optimization workloads from the OverlayCCL/Sorcar campaign: every registered problem with a candidate file,
#    captured as baseline template / Sorcar candidate / strat candidate at 2 ranks

ACC_REPO = os.environ.get("ACC_REPO", "/home/ubuntu/acc") if "os" in globals() else "/home/ubuntu/acc"

# ---------------------------------------------------------------------------- new builders
def b_synth_norm(p, dev):
    """normalization-heavy stacks: layernorm / rmsnorm / softmax / batchnorm-like statistics over odd last dims"""
    B, T, D = p["batch"], p["seq"], p["dim"]; x = torch.randn(B, T, D, device=dev); w = torch.randn(D, device=dev); kind = p["kind"]; n = p["n_ops"]
    def step():
        a = x
        for i in range(n):
            if kind == "layernorm": a = torch.nn.functional.layer_norm(a, (D,), weight=w) * 1.01
            elif kind == "rmsnorm": a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-6) * w
            elif kind == "softmax": a = torch.softmax(a * 1.1, -1) + a
            elif kind == "stats": a = (a - a.mean(1, keepdim=True)) / (a.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            elif kind == "mixed": a = torch.softmax(torch.nn.functional.layer_norm(a, (D,)), -1) + a * w
        return [a]
    return step, []

def b_synth_layout(p, dev):
    """layout-changing chains on rank-3/4 tensors with non-aligned dims: permute, transpose, reshape, flatten, unflatten"""
    x = torch.randn(*p["shape"], device=dev); rng = random.Random(p["seed"]); n = p["n_ops"]; r = len(p["shape"])
    plan = [rng.choice(["perm", "tr", "flat", "unflat", "mm"]) for _ in range(n)]
    def step():
        a = x
        for op in plan:
            if op == "perm":
                dims = list(range(a.dim())); rng2 = random.Random(len(plan) + a.dim()); rng2.shuffle(dims); a = a.permute(*dims).contiguous()
            elif op == "tr": a = a.transpose(0, -1).contiguous()
            elif op == "flat": a = a.reshape(a.shape[0], -1)
            elif op == "unflat": a = a.reshape(*x.shape) if a.numel() == x.numel() else a
            elif op == "mm": a = a.reshape(-1, a.shape[-1]) @ torch.ones(a.shape[-1], a.shape[-1], device=dev) if a.shape[-1] <= 4096 else a
        return [a]
    return step, []

def b_synth_slice(p, dev):
    """static slice / pad / concat / roll / narrow chains with odd offsets (dynamic-slice and pad lowering)"""
    x = torch.randn(*p["shape"], device=dev); n = p["n_ops"]; off = p["offset"]
    def step():
        a = x
        for i in range(n):
            k = i % 5
            if k == 0: a = torch.nn.functional.pad(a, (off, off))
            elif k == 1: a = a[..., off:a.shape[-1] - off] if a.shape[-1] > 2 * off + 1 else a
            elif k == 2: a = torch.cat([a[..., off:], a[..., :off]], -1)
            elif k == 3: a = torch.roll(a, off, -1)
            elif k == 4: a = a.narrow(0, 0, max(1, a.shape[0] - 1)) if a.shape[0] > 1 else a
        return [a]
    return step, []

def b_synth_mask(p, dev):
    """masking / select / boolean reductions: where, masked_fill, nonzero-free masks, compare chains, int arithmetic"""
    x = torch.randn(*p["shape"], device=dev); m = torch.rand(*p["shape"], device=dev) > 0.5; n = p["n_ops"]; kind = p["kind"]
    ii = torch.randint(0, 1000, p["shape"], device=dev)
    def step():
        a = x; b = ii
        for i in range(n):
            if kind == "where": a = torch.where(m, a * 2, a - 1)
            elif kind == "masked_fill": a = a.masked_fill(m ^ (a > 0), 0.5)
            elif kind == "bool_reduce": a = a + (m & (a > 0)).sum(-1, keepdim=True).float()
            elif kind == "int_ops": b = (b * 3 + 7) % 1009; a = a + b.float()
            elif kind == "cmp_chain": a = (a > 0).float() * a + (a <= 0).float() * torch.exp(torch.clamp(a, -5, 5))
        return [a, b]
    return step, []

def b_synth_scatter(p, dev):
    """index_put / scatter / scatter_reduce / index_copy with random or sorted indices"""
    V, D, N = p["rows"], p["dim"], p["n_idx"]; table = torch.randn(V, D, device=dev); idx = torch.randint(0, V, (N,), device=dev)
    if p.get("sorted"): idx = torch.sort(idx).values
    src = torch.randn(N, D, device=dev); kind = p["kind"]
    def step():
        if kind == "index_put": t = table.clone(); t[idx] = src; return [t]
        if kind == "index_put_acc": t = table.clone(); t.index_put_((idx,), src, accumulate=True); return [t]
        if kind == "scatter": return [table.scatter(0, idx.unsqueeze(1).expand(N, D), src)]
        if kind == "scatter_reduce_amax": return [table.scatter_reduce(0, idx.unsqueeze(1).expand(N, D), src, "amax")]
        if kind == "index_copy": return [table.index_copy(0, idx[: min(N, V)], src[: min(N, V)])] if p.get("sorted") else [table.index_copy(0, torch.arange(min(N, V), device=dev), src[: min(N, V)])]
        if kind == "embedding_bag": return [torch.nn.functional.embedding_bag(idx.view(-1, 8), table, mode="sum")]
        raise ValueError(kind)
    return step, []

def b_synth_conv2(p, dev):
    """convolution variants: depthwise, dilated, strided, grouped, pooling, upsample, 1-D conv, conv-transpose"""
    B, C, H = p["batch"], p["ch"], p["img"]; kind = p["kind"]; x = torch.randn(B, C, H, H, device=dev)
    if kind == "depthwise": m = torch.nn.Conv2d(C, C, 3, padding=1, groups=C)
    elif kind == "dilated": m = torch.nn.Conv2d(C, C, 3, padding=2, dilation=2)
    elif kind == "strided": m = torch.nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
    elif kind == "grouped": m = torch.nn.Conv2d(C, C, 3, padding=1, groups=max(1, C // 8))
    elif kind == "k1": m = torch.nn.Conv2d(C, 4 * C, 1)
    elif kind == "k5": m = torch.nn.Conv2d(C, C, 5, padding=2)
    elif kind == "k7_stride2": m = torch.nn.Conv2d(C, C, 7, stride=2, padding=3)
    elif kind == "convT": m = torch.nn.ConvTranspose2d(C, C, 4, stride=2, padding=1)
    elif kind == "pool_avg": m = torch.nn.Sequential(torch.nn.Conv2d(C, C, 3, padding=1), torch.nn.AvgPool2d(2), torch.nn.Conv2d(C, C, 3, padding=1))
    elif kind == "pool_max": m = torch.nn.Sequential(torch.nn.Conv2d(C, C, 3, padding=1), torch.nn.MaxPool2d(3, 2, 1))
    elif kind == "upsample": m = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode="nearest"), torch.nn.Conv2d(C, C, 3, padding=1))
    elif kind == "conv1d":
        m = torch.nn.Conv1d(C, C, 5, padding=2); x = torch.randn(B, C, H * H, device=dev)
    else: raise ValueError(kind)
    m = m.to(dev)
    if p.get("train"): return _train_step(m, x, lambda o: (o ** 2).mean())
    return (lambda: [m(x)]), list(m.parameters())

def b_synth_einsum(p, dev):
    """einsum / batched-matmul shapes with uneven contraction dims and broadcasting"""
    kind = p["kind"]; a_, b_, c_, d_ = p["dims"]
    A = torch.randn(a_, b_, c_, device=dev); Bm = torch.randn(a_, c_, d_, device=dev); C = torch.randn(c_, d_, device=dev); Dm = torch.randn(b_, c_, device=dev)
    def step():
        if kind == "bmm": return [torch.bmm(A, Bm)]
        if kind == "bik_kj": return [torch.einsum("bik,kj->bij", A, C)]
        if kind == "bik_bkj_sum": return [torch.einsum("bik,bkj->ij", A, Bm)]
        if kind == "outer_batch": return [torch.einsum("bi,bj->bij", A[:, :, 0], Bm[:, 0, :])]
        if kind == "trace_like": return [torch.einsum("bii->b", A[:, : min(b_, c_), : min(b_, c_)])]
        if kind == "bik_ik": return [torch.einsum("bik,ik->bi", A, Dm)]
        if kind == "chain3": return [torch.einsum("bik,kj,jl->bil", A, C, torch.randn(d_, b_, device=dev))]
        if kind == "mm_T": return [A.transpose(1, 2) @ A]
        raise ValueError(kind)
    return step, []

def b_synth_rnn(p, dev):
    """unrolled recurrence: elman / gru-like cell over T steps (sequential dependency chain, no fusion across steps)"""
    B, T, D, Hd = p["batch"], p["seq"], p["dim"], p["hidden"]; cell = p["cell"]
    Wx = torch.randn(D, Hd, device=dev) * 0.1; Wh = torch.randn(Hd, Hd, device=dev) * 0.1; Wz = torch.randn(D + Hd, Hd, device=dev) * 0.1
    x = torch.randn(B, T, D, device=dev)
    def step():
        h = torch.zeros(B, Hd, device=dev); outs = []
        for t in range(T):
            xt = x[:, t]
            if cell == "elman": h = torch.tanh(xt @ Wx + h @ Wh)
            elif cell == "gru_like":
                z = torch.sigmoid(torch.cat([xt, h], -1) @ Wz); h = z * h + (1 - z) * torch.tanh(xt @ Wx + h @ Wh)
            elif cell == "cumsum": h = h + xt @ Wx
            outs.append(h)
        return [torch.stack(outs, 1)]
    return step, []

def b_synth_logits(p, dev):
    """large-vocab projection + cross-entropy (+ backward): the memory-heavy tail of LM graphs"""
    B, T, D, V = p["batch"], p["seq"], p["dim"], p["vocab"]; dt = _dtype(p.get("dtype", "fp32"))
    h = torch.randn(B, T, D, device=dev, dtype=dt); W = torch.randn(D, V, device=dev, dtype=dt, requires_grad=True); y = torch.randint(0, V, (B, T), device=dev)
    def step():
        logits = (h @ W).float().view(-1, V); loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
        if p.get("train"):
            (g,) = torch.autograd.grad(loss, [W]); return [loss, g]
        return [loss]
    return step, [W]

def b_synth_cast(p, dev):
    """dtype-cast chains: fp32 <-> bf16 <-> fp16 <-> int around matmuls and reductions"""
    n = p["n_ops"]; x = torch.randn(*p["shape"], device=dev); w = torch.randn(p["shape"][-1], p["shape"][-1], device=dev)
    def step():
        a = x
        for i in range(n):
            k = i % 4
            if k == 0: a = (a.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()
            elif k == 1: a = (a.to(torch.float16) * 1.5).float()
            elif k == 2: a = a.round().to(torch.int32).to(torch.float32) * 0.5 + a
            elif k == 3: a = (a.to(torch.bfloat16).sum(-1, keepdim=True)).float() + a
        return [a]
    return step, []

def b_acc_cand(p, dev):
    """collective-optimization workload from the OverlayCCL/Sorcar campaign, run at the current world size under torchrun.
    p = {problem, variant: builtin | sorcar | strat}; the candidate code comes from the campaign's result files."""
    import glob, importlib, importlib.util, sys, types, os as _os
    import torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    if ACC_REPO not in sys.path: sys.path.insert(0, ACC_REPO)
    import search.problems as P
    if len(P.PROBLEMS) < 100:
        for mp in sorted(glob.glob(_os.path.join(ACC_REPO, "search", "problems_*.py"))):
            try: importlib.import_module("search." + _os.path.splitext(_os.path.basename(mp))[0])
            except Exception: pass
    prob = P.get_problem(p["problem"]); variant = p["variant"]
    if variant == "builtin": code = list(prob.builtin_templates.values())[0]
    else: code = open(_os.path.join(ACC_REPO, "taxonomy_3col_results", f"{variant}_code", f"{p['problem']}.py")).read()
    mod = types.ModuleType(f"cand_{p['problem']}_{variant}"); mod.__dict__.update(dict(torch=torch, xm=xm, math=math)); exec(code, mod.__dict__)
    fn = getattr(mod, prob.evolved_fn_name, None)
    if fn is None:
        fn = next(getattr(mod, nm) for nm in dir(mod) if (nm.endswith("_fn") or nm.startswith("evolved_")) and callable(getattr(mod, nm)))
    world = xr.world_size(); rank = xr.global_ordinal(); cpd = 2; nd = max(1, world // cpd)
    torch.manual_seed(1234); tc = prob.generate_test_case(world); pra = tc["per_rank_args"][rank]; shared = tc.get("shared_args", {})
    def to_dev(v):
        if isinstance(v, torch.Tensor): return v.to(dev)
        if isinstance(v, (list, tuple)): return type(v)(to_dev(u) for u in v)
        if isinstance(v, dict): return {k: to_dev(u) for k, u in v.items()}
        return v
    args = to_dev(pra); shared = to_dev(shared)
    keep = [v for v in args.values() if isinstance(v, torch.Tensor)]
    def step():
        out = prob.call_candidate(fn, args, shared, rank, world, nd, cpd, xm, torch, num_nodes=1)
        if isinstance(out, torch.Tensor): return [out]
        if isinstance(out, (list, tuple)): return [o for o in out if isinstance(o, torch.Tensor)]
        return []
    return step, keep

BUILDERS.update(synth_norm=b_synth_norm, synth_layout=b_synth_layout, synth_slice=b_synth_slice, synth_mask=b_synth_mask, synth_scatter=b_synth_scatter,
                synth_conv2=b_synth_conv2, synth_einsum=b_synth_einsum, synth_rnn=b_synth_rnn, synth_logits=b_synth_logits, synth_cast=b_synth_cast, acc_cand=b_acc_cand)

# ---------------------------------------------------------------------------- new grids
def _sweep(base, key, values, fix=None):
    out = []
    for v in values:
        q = dict(base["params"]); q[key] = v
        if fix: q = fix(q)
        if q is None: continue
        s = _spec(base["family"], "sweep", **q); s["id"] = f"sweep__{base['id']}__{key}{v}"; s["parent"] = base["id"]; s["sweep_key"] = key; s["sweep_value"] = v; out.append(s)
    return out

def grid_sweeps():
    """one-dimensional sweeps through boundary regions: ±k around 128/512/1024/2048/4096, small batches, head/layer counts,
    the gather-table and randn-size cliffs, the conv batch failure"""
    S = []
    mlp = _spec("synth_mlp", "synth", d=512, hidden=2048, depth=4, batch=1024, act="gelu", train=True)
    S += _sweep(mlp, "hidden", [2040, 2044, 2046, 2047, 2048, 2049, 2050, 2052, 2056, 2064, 2080, 2112, 2176, 2304, 2560, 3072, 3584, 4095, 4096, 4097, 4100])
    S += _sweep(mlp, "batch", [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048])
    S += _sweep(mlp, "d", [120, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 640, 767, 768, 769, 1023, 1024, 1025])
    S += _sweep(mlp, "depth", [1, 2, 3, 5, 6, 8, 12, 16, 24, 32])
    attn = _spec("synth_attn", "synth", dim=1024, heads=8, layers=2, seq=1024, batch=4, train=True)
    S += _sweep(attn, "seq", [120, 127, 128, 129, 130, 192, 255, 256, 257, 384, 511, 512, 513, 640, 768, 1000, 1023, 1024, 1025, 1026, 1152, 1536, 2047, 2048, 2049])
    S += _sweep(attn, "batch", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 32, 33])
    S += _sweep(attn, "heads", [1, 2, 4, 8, 16, 32, 64, 128], fix=lambda q: q if q["dim"] % q["heads"] == 0 else None)
    S += _sweep(attn, "dim", [768, 1024, 1152, 1280, 1536, 2048, 2176, 2560, 3072, 4096], fix=lambda q: q if q["dim"] % q["heads"] == 0 else None)
    S += _sweep(attn, "layers", [1, 2, 3, 4, 6, 8, 12])
    gpt = _spec("hf", "real", arch="gpt2", layers=2, hidden=768, heads=12, batch=2, seq=512, train=True)
    S += _sweep(gpt, "batch", [1, 2, 3, 4, 5, 6, 8, 12, 16])
    S += _sweep(gpt, "seq", [64, 100, 127, 128, 129, 192, 256, 384, 511, 512, 513, 640, 768, 1024])
    S += _sweep(gpt, "hidden", [768, 769, 772, 776, 784, 800, 832, 896, 960, 1024, 1536, 2048], fix=lambda q: (dict(q, heads=q["hidden"] // 64) if q["hidden"] % 64 == 0 else dict(q, heads=8) if q["hidden"] % 8 == 0 else None))
    S += _sweep(gpt, "layers", [1, 2, 3, 4, 6, 8, 12, 16])
    S += _sweep(gpt, "vocab", [8192, 16384, 32000, 32768, 50257, 50304, 65536, 100000, 128000, 151936])
    llama = _spec("hf", "real", arch="llama", layers=2, hidden=1024, heads=8, batch=1, seq=512, train=True)
    S += _sweep(llama, "batch", [1, 2, 3, 4, 6, 8])
    S += _sweep(llama, "seq", [128, 256, 511, 512, 513, 1024, 2048, 4096])
    S += _sweep(llama, "kv_heads", [1, 2, 4, 8])
    S += _sweep(llama, "inter", [2048, 2730, 2752, 3072, 4096, 5504, 8192, 11008])
    conv = _spec("synth_conv", "synth", ch=128, layers=8, img=64, batch=8, train=True)
    S += _sweep(conv, "batch", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 32])
    S += _sweep(conv, "img", [16, 24, 28, 31, 32, 33, 48, 56, 63, 64, 65, 96, 112, 127, 128, 129, 160, 224, 256])
    S += _sweep(conv, "ch", [8, 16, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 255, 256, 257, 384, 512])
    gather = _spec("synth_gather", "synth", kind="gather", vocab=1024, dim=256, n_idx=8192)
    S += _sweep(gather, "vocab", [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 65536])
    S += _sweep(gather, "n_idx", [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    S += _sweep(gather, "dim", [1, 8, 32, 64, 127, 128, 129, 256, 512, 1024, 2048])
    isel = _spec("synth_gather", "synth", kind="index_select", vocab=1024, dim=256, n_idx=8192)
    S += _sweep(isel, "vocab", [1024, 8192, 32768, 65536, 131072, 262144])
    ew = _spec("synth_elementwise", "synth", n_ops=8, numel=1_000_000, seed=0)
    S += _sweep(ew, "numel", [100_000, 250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000, 12_000_000, 16_000_000])
    S += _sweep(ew, "n_ops", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    msm = _spec("synth_many_small", "synth", n_mats=128, mat=64, mode="unrolled")
    S += _sweep(msm, "n_mats", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    S += _sweep(msm, "mat", [8, 16, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 256, 512])
    return S

def grid_synth2():
    S = []
    for kind, dim, n_ops in itertools.product(["layernorm", "rmsnorm", "softmax", "stats", "mixed"], [1024, 1000, 4096], [4, 32]):
        S.append(_spec("synth_norm", "synth", kind=kind, batch=8, seq=512, dim=dim, n_ops=n_ops))
    for shape, n_ops, seed in itertools.product(([8, 96, 1000], [2, 3, 1024, 1024], [12, 7, 128, 384]), [8, 32], [0, 1]):
        S.append(_spec("synth_layout", "synth", shape=shape, n_ops=n_ops, seed=seed))
    for shape, n_ops, off in itertools.product(([4096, 4096], [16, 1024, 1024], [1000000]), [8, 32], [1, 7, 64]):
        S.append(_spec("synth_slice", "synth", shape=shape, n_ops=n_ops, offset=off))
    for kind, shape, n_ops in itertools.product(["where", "masked_fill", "bool_reduce", "int_ops", "cmp_chain"], ([4096, 4096], [1000000]), [8, 64]):
        S.append(_spec("synth_mask", "synth", kind=kind, shape=shape, n_ops=n_ops))
    for kind, rows, n_idx, srt in itertools.product(["index_put", "index_put_acc", "scatter", "scatter_reduce_amax", "index_copy", "embedding_bag"], [1024, 32768], [1024, 16384], [False, True]):
        if kind == "index_copy" and not srt: continue
        S.append(_spec("synth_scatter", "synth", kind=kind, rows=rows, dim=256, n_idx=n_idx, sorted=srt))
    for kind, ch, img, train in itertools.product(["depthwise", "dilated", "strided", "grouped", "k1", "k5", "k7_stride2", "convT", "pool_avg", "pool_max", "upsample", "conv1d"], [64], [64], [False, True]):
        S.append(_spec("synth_conv2", "synth", kind=kind, batch=8, ch=ch, img=img, train=train))
    for kind, dims in itertools.product(["bmm", "bik_kj", "bik_bkj_sum", "outer_batch", "trace_like", "bik_ik", "chain3", "mm_T"], ([8, 512, 512, 512], [64, 128, 1000, 96], [1, 4096, 64, 4096])):
        S.append(_spec("synth_einsum", "synth", kind=kind, dims=dims))
    for cell, seq, hidden in itertools.product(["elman", "gru_like", "cumsum"], [16, 64, 256], [256, 1024]):
        S.append(_spec("synth_rnn", "synth", cell=cell, batch=32, seq=seq, dim=256, hidden=hidden))
    for vocab, seq, train, dt in itertools.product([32000, 50257, 128000], [512, 2048], [False, True], ["fp32", "bf16"]):
        S.append(_spec("synth_logits", "synth", batch=2, seq=seq, dim=1024, vocab=vocab, train=train, dtype=dt))
    for shape, n_ops in itertools.product(([1024, 1024], [8, 512, 2048]), [4, 16, 64]):
        S.append(_spec("synth_cast", "synth", shape=shape, n_ops=n_ops))
    return S

def grid_acc_cand():
    """all campaign problems with a Sorcar/strat candidate file: baseline template + Sorcar + strat (run at 2 ranks)"""
    import glob, os as _os
    S = []
    probs = sorted(_os.path.splitext(_os.path.basename(f))[0] for f in glob.glob(_os.path.join(ACC_REPO, "taxonomy_3col_results", "sorcar_code", "*.py")))
    for pr in probs:
        for variant in ("builtin", "sorcar", "strat"):
            if variant == "strat" and not _os.path.exists(_os.path.join(ACC_REPO, "taxonomy_3col_results", "strat_code", f"{pr}.py")): continue
            S.append({"id": f"acc_cand__{pr}__{variant}", "family": "acc_cand", "source": "collective", "params": {"problem": pr, "variant": variant}, "n_steps": 3})
    return S

GRIDS.update(sweeps=grid_sweeps, synth2=grid_synth2, acc_cand=grid_acc_cand)


# ============================================================================= Step 1b: real workloads (2026-10-23)
# more model families from HF configs (random init), torchvision classification models, realistic training steps
# (AdamW + grad clipping + bf16 autocast + label smoothing + gradient accumulation), recommendation (DLRM-like),
# graph message passing, dense-routed MoE, a diffusion-style UNet block, two-tower contrastive training.

def b_hf2(p, dev):
    import transformers as T
    arch = p["arch"]; dt = _dtype(p.get("dtype", "fp32")); B, S = p["batch"], p.get("seq", 128); L = p["layers"]; H = p["hidden"]; nh = p["heads"]; train = bool(p.get("train"))
    lm = "causal"; vocab = p.get("vocab", 32000)
    if arch == "mistral": cfg = T.MistralConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, num_key_value_heads=p.get("kv_heads", nh), intermediate_size=p.get("inter", 4 * H), vocab_size=vocab, max_position_embeddings=max(S, 128), sliding_window=p.get("window", 4096), use_cache=False); m = T.MistralForCausalLM(cfg)
    elif arch == "qwen2": cfg = T.Qwen2Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, num_key_value_heads=p.get("kv_heads", nh), intermediate_size=p.get("inter", 4 * H), vocab_size=vocab, max_position_embeddings=max(S, 128), use_cache=False); m = T.Qwen2ForCausalLM(cfg)
    elif arch == "phi3": cfg = T.Phi3Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, num_key_value_heads=p.get("kv_heads", nh), intermediate_size=p.get("inter", 4 * H), vocab_size=vocab, max_position_embeddings=max(S, 128), use_cache=False, pad_token_id=0, eos_token_id=2, bos_token_id=1); m = T.Phi3ForCausalLM(cfg)
    elif arch == "gemma2": cfg = T.Gemma2Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, num_key_value_heads=p.get("kv_heads", nh), intermediate_size=p.get("inter", 4 * H), vocab_size=vocab, max_position_embeddings=max(S, 128), head_dim=H // nh, use_cache=False); m = T.Gemma2ForCausalLM(cfg)
    elif arch == "opt": cfg = T.OPTConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, ffn_dim=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128), word_embed_proj_dim=H, use_cache=False); m = T.OPTForCausalLM(cfg)
    elif arch == "bloom": cfg = T.BloomConfig(n_layer=L, hidden_size=H, n_head=nh, vocab_size=vocab, use_cache=False); m = T.BloomForCausalLM(cfg)
    elif arch == "falcon": cfg = T.FalconConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, vocab_size=vocab, use_cache=False, new_decoder_architecture=True, num_kv_heads=p.get("kv_heads", nh)); m = T.FalconForCausalLM(cfg)
    elif arch == "gpt_neox": cfg = T.GPTNeoXConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128), use_cache=False); m = T.GPTNeoXForCausalLM(cfg)
    elif arch == "gptj": cfg = T.GPTJConfig(n_layer=L, n_embd=H, n_head=nh, vocab_size=vocab, n_positions=max(S, 128), rotary_dim=min(64, H // nh), use_cache=False); m = T.GPTJForCausalLM(cfg)
    elif arch == "roberta": cfg = T.RobertaConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128) + 2); m = T.RobertaForMaskedLM(cfg); lm = "mlm"
    elif arch == "distilbert": cfg = T.DistilBertConfig(n_layers=L, dim=H, n_heads=nh, hidden_dim=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128)); m = T.DistilBertForMaskedLM(cfg); lm = "mlm"
    elif arch == "albert": cfg = T.AlbertConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, embedding_size=128, vocab_size=vocab, max_position_embeddings=max(S, 128)); m = T.AlbertForMaskedLM(cfg); lm = "mlm"
    elif arch == "electra": cfg = T.ElectraConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, embedding_size=128, vocab_size=vocab, max_position_embeddings=max(S, 128)); m = T.ElectraForMaskedLM(cfg); lm = "mlm"
    elif arch == "deberta_v2": cfg = T.DebertaV2Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128), relative_attention=True, position_buckets=64); m = T.DebertaV2ForMaskedLM(cfg); lm = "mlm"
    elif arch == "xlm_roberta": cfg = T.XLMRobertaConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128) + 2); m = T.XLMRobertaForMaskedLM(cfg); lm = "mlm"
    elif arch == "bart": cfg = T.BartConfig(encoder_layers=L, decoder_layers=L, d_model=H, encoder_attention_heads=nh, decoder_attention_heads=nh, encoder_ffn_dim=4 * H, decoder_ffn_dim=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128), use_cache=False); m = T.BartForConditionalGeneration(cfg); lm = "seq2seq"
    elif arch == "whisper": cfg = T.WhisperConfig(encoder_layers=L, decoder_layers=L, d_model=H, encoder_attention_heads=nh, decoder_attention_heads=nh, encoder_ffn_dim=4 * H, decoder_ffn_dim=4 * H, num_mel_bins=80, max_source_positions=p.get("frames", 1500) // 2, max_target_positions=max(S, 128), vocab_size=vocab, use_cache=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[]); m = T.WhisperForConditionalGeneration(cfg); lm = "whisper"
    elif arch == "wav2vec2": cfg = T.Wav2Vec2Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=64); m = T.Wav2Vec2ForCTC(cfg); lm = "ctc"
    elif arch == "clip": cfg = T.CLIPConfig(text_config=dict(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, vocab_size=vocab, max_position_embeddings=max(S, 128)), vision_config=dict(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, image_size=p.get("img", 224), patch_size=p.get("patch", 32)), projection_dim=H); m = T.CLIPModel(cfg); lm = "clip"
    elif arch == "dinov2": cfg = T.Dinov2Config(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, intermediate_size=4 * H, image_size=p.get("img", 224), patch_size=p.get("patch", 14), num_labels=1000); m = T.Dinov2ForImageClassification(cfg); lm = "img"
    elif arch == "convnext": cfg = T.ConvNextConfig(depths=p.get("depths", [1, 1, L, 1]), hidden_sizes=[H // 8, H // 4, H // 2, H], num_labels=1000); m = T.ConvNextForImageClassification(cfg); lm = "img"
    elif arch == "swin": cfg = T.SwinConfig(depths=p.get("depths", [1, 1, L, 1]), embed_dim=H // 8, num_heads=[nh // 8 or 1, nh // 4 or 1, nh // 2 or 1, nh], image_size=p.get("img", 224), num_labels=1000); m = T.SwinForImageClassification(cfg); lm = "img"
    elif arch == "resnet": cfg = T.ResNetConfig(depths=p.get("depths", [L, L, L, L]), hidden_sizes=[H // 8, H // 4, H // 2, H], embedding_size=64, layer_type=p.get("layer_type", "bottleneck"), num_labels=1000); m = T.ResNetForImageClassification(cfg); lm = "img"
    elif arch == "mobilenet_v2": cfg = T.MobileNetV2Config(image_size=p.get("img", 224), depth_multiplier=p.get("width", 1.0), num_labels=1000); m = T.MobileNetV2ForImageClassification(cfg); lm = "img"
    elif arch == "segformer": cfg = T.SegformerConfig(depths=[1, 1, L, 1], hidden_sizes=[H // 8, H // 4, H // 2, H], num_attention_heads=[1, 2, 4, 8], num_labels=19); m = T.SegformerForSemanticSegmentation(cfg); lm = "seg"
    else: raise ValueError(arch)
    m = m.to(dev, dt); m.train(train)
    V = getattr(cfg, "vocab_size", vocab) if lm not in ("clip",) else cfg.text_config.vocab_size
    if lm in ("causal", "mlm"):
        ids = torch.randint(0, V, (B, S), device=dev); fwd = lambda: m(input_ids=ids, labels=ids)
    elif lm == "seq2seq":
        ids = torch.randint(0, V, (B, S), device=dev); fwd = lambda: m(input_ids=ids, labels=ids)
    elif lm == "whisper":
        feats = torch.randn(B, 80, p.get("frames", 1500), device=dev, dtype=dt); ids = torch.randint(0, V, (B, S), device=dev); fwd = lambda: m(input_features=feats, labels=ids)
    elif lm == "ctc":
        wav = torch.randn(B, p.get("samples", 16000), device=dev, dtype=dt); lab = torch.randint(1, cfg.vocab_size - 1, (B, 32), device=dev); fwd = lambda: m(input_values=wav, labels=lab)
    elif lm == "clip":
        ids = torch.randint(0, V, (B, S), device=dev); img = torch.randn(B, 3, p.get("img", 224), p.get("img", 224), device=dev, dtype=dt); fwd = lambda: m(input_ids=ids, pixel_values=img, return_loss=True)
    elif lm == "img":
        img = torch.randn(B, 3, p.get("img", 224), p.get("img", 224), device=dev, dtype=dt); lab = torch.randint(0, 1000, (B,), device=dev); fwd = lambda: m(pixel_values=img, labels=lab)
    elif lm == "seg":
        img = torch.randn(B, 3, p.get("img", 128), p.get("img", 128), device=dev, dtype=dt); lab = torch.randint(0, 19, (B, p.get("img", 128), p.get("img", 128)), device=dev); fwd = lambda: m(pixel_values=img, labels=lab)
    params = [q for q in m.parameters() if q.requires_grad]   # e.g. Whisper's sinusoidal positions are frozen buffers-as-parameters
    if train:
        def step():
            loss = fwd().loss.float(); grads = torch.autograd.grad(loss, params, allow_unused=True)
            with torch.no_grad():
                for q, g in zip(params, grads):
                    if g is not None: q.sub_(1e-3 * g)
            return [loss]
        return step, params
    def step():
        out = fwd(); o = getattr(out, "logits", None)
        if o is None: o = getattr(out, "loss", None)
        if o is None: o = out[0]
        return [o]
    return step, params

def b_tv(p, dev):
    """torchvision classification model, random init"""
    import torchvision
    m = torchvision.models.get_model(p["name"], weights=None).to(dev); m.train(bool(p.get("train")))
    x = torch.randn(p["batch"], 3, p.get("img", 224), p.get("img", 224), device=dev); y = torch.randint(0, 1000, (p["batch"],), device=dev)
    if p.get("train"): return _train_step(m, x, lambda o: torch.nn.functional.cross_entropy(o, y))
    return (lambda: [m(x)]), list(m.parameters())

def b_train_real(p, dev):
    """a realistic training step: AdamW optimizer, gradient clipping, optional bf16 autocast, label smoothing, gradient
    accumulation over several micro-batches (all inside one traced graph)"""
    import transformers as T
    H, nh, L, B, S = p["hidden"], p["heads"], p["layers"], p["batch"], p["seq"]
    if p["arch"] == "gpt2": m = T.GPT2LMHeadModel(T.GPT2Config(n_layer=L, n_embd=H, n_head=nh, n_positions=max(S, 64), vocab_size=p.get("vocab", 50257), use_cache=False))
    else: m = T.LlamaForCausalLM(T.LlamaConfig(num_hidden_layers=L, hidden_size=H, num_attention_heads=nh, num_key_value_heads=p.get("kv_heads", nh), intermediate_size=4 * H, vocab_size=p.get("vocab", 32000), max_position_embeddings=max(S, 64), use_cache=False))
    m = m.to(dev); m.train()
    opt = {"adamw": lambda ps: torch.optim.AdamW(ps, lr=1e-4, weight_decay=0.1), "sgd_momentum": lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9), "adagrad": lambda ps: torch.optim.Adagrad(ps, lr=1e-2)}[p.get("optimizer", "adamw")](m.parameters())
    ids = [torch.randint(0, m.config.vocab_size, (B, S), device=dev) for _ in range(p.get("accum", 1))]
    ls = p.get("label_smoothing", 0.0); clip = p.get("clip", 1.0); ac = p.get("autocast")
    def step():
        opt.zero_grad(set_to_none=True); tot = 0.0
        for x in ids:
            with torch.autocast("xla" if "xla" in str(dev) else "cpu", dtype=torch.bfloat16, enabled=bool(ac)):
                logits = m(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1), label_smoothing=ls) / len(ids)
            loss.backward(); tot = tot + loss.detach()
        if clip: torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        return [tot]
    return step, list(m.parameters())

def b_dlrm(p, dev):
    """recommendation model: embedding tables + bottom MLP + pairwise dot interaction + top MLP + BCE, trained"""
    nt, rows, d, B = p["n_tables"], p["rows"], p["dim"], p["batch"]
    tables = torch.nn.ModuleList([torch.nn.EmbeddingBag(rows, d, mode="sum") for _ in range(nt)]).to(dev)
    bot = MLP(p["dense"], 256, 2, "relu").to(dev); proj = torch.nn.Linear(p["dense"], d).to(dev)
    n_feat = nt + 1; top = torch.nn.Sequential(torch.nn.Linear(n_feat * (n_feat - 1) // 2 + d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    dense = torch.randn(B, p["dense"], device=dev); idx = [torch.randint(0, rows, (B * p["lookups"],), device=dev) for _ in range(nt)]
    offs = torch.arange(0, B * p["lookups"], p["lookups"], device=dev); y = torch.rand(B, 1, device=dev).round()
    model = torch.nn.ModuleList([tables, bot, proj, top])
    def loss_fn():
        x = proj(bot(dense)); embs = [t(i, offs) for t, i in zip(tables, idx)]; feats = torch.stack([x] + embs, 1)
        inter = torch.bmm(feats, feats.transpose(1, 2)); iu = torch.triu_indices(n_feat, n_feat, 1, device=dev); flat = inter[:, iu[0], iu[1]]
        out = top(torch.cat([x, flat], 1)); return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    params = list(model.parameters())
    def step():
        loss = loss_fn(); grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for q, g in zip(params, grads):
                if g is not None: q.sub_(1e-3 * g)
        return [loss]
    return step, params

def b_gnn(p, dev):
    """graph message passing (GCN-style) with scatter_add aggregation over a random edge list"""
    N, E, D, L = p["nodes"], p["edges"], p["dim"], p["layers"]
    x = torch.randn(N, D, device=dev); src = torch.randint(0, N, (E,), device=dev); dst = torch.randint(0, N, (E,), device=dev)
    Ws = [torch.randn(D, D, device=dev) * 0.1 for _ in range(L)]; deg = torch.zeros(N, device=dev).index_add(0, dst, torch.ones(E, device=dev)).clamp(min=1).unsqueeze(1)
    def step():
        h = x
        for W in Ws:
            msg = h[src] @ W; agg = torch.zeros(N, D, device=dev).index_add(0, dst, msg) / deg; h = torch.relu(agg + h)
        return [h.sum(0)]
    return step, []

def b_moe(p, dev):
    """dense-routed mixture of experts: top-k gating, all experts computed, outputs masked (the XLA-friendly formulation)"""
    B, S, D, E, k = p["batch"], p["seq"], p["dim"], p["experts"], p["topk"]; Fd = p.get("ffn", 4 * D)
    x = torch.randn(B * S, D, device=dev); gate = torch.randn(D, E, device=dev) * 0.1
    W1 = torch.randn(E, D, Fd, device=dev) * 0.05; W2 = torch.randn(E, Fd, D, device=dev) * 0.05
    def step():
        logits = x @ gate; top_vals, top_idx = torch.topk(logits, k, -1); probs = torch.softmax(top_vals, -1)
        mask = torch.zeros(B * S, E, device=dev).scatter(1, top_idx, probs)
        h = torch.einsum("td,edf->tef", x, W1); h = torch.nn.functional.gelu(h); y = torch.einsum("tef,efd->ted", h, W2)
        out = (y * mask.unsqueeze(-1)).sum(1)
        aux = (mask.mean(0) * torch.softmax(logits, -1).mean(0)).sum() * E
        return [out, aux]
    return step, []

def b_unet_block(p, dev):
    """diffusion-style UNet stage: resnet blocks (conv + groupnorm + SiLU + timestep embedding) + self-attention, trained"""
    C, Hs, B = p["ch"], p["img"], p["batch"]
    class Res(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.n1 = torch.nn.GroupNorm(8, C); s.c1 = torch.nn.Conv2d(C, C, 3, padding=1); s.t = torch.nn.Linear(128, C); s.n2 = torch.nn.GroupNorm(8, C); s.c2 = torch.nn.Conv2d(C, C, 3, padding=1)
        def forward(s, x, temb): h = s.c1(torch.nn.functional.silu(s.n1(x))) + s.t(temb)[:, :, None, None]; return x + s.c2(torch.nn.functional.silu(s.n2(h)))
    class Attn2d(torch.nn.Module):
        def __init__(s): super().__init__(); s.n = torch.nn.GroupNorm(8, C); s.qkv = torch.nn.Linear(C, 3 * C); s.o = torch.nn.Linear(C, C)
        def forward(s, x):
            b, c, h, w = x.shape; t = s.n(x).flatten(2).transpose(1, 2); q, k, v = s.qkv(t).chunk(3, -1)
            a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(c), -1) @ v; return x + s.o(a).transpose(1, 2).reshape(b, c, h, w)
    class Stage(torch.nn.Module):
        def __init__(s): super().__init__(); s.inp = torch.nn.Conv2d(3, C, 3, padding=1); s.r1 = Res(); s.a = Attn2d() if p.get("attn", True) else torch.nn.Identity(); s.r2 = Res(); s.down = torch.nn.Conv2d(C, C, 3, stride=2, padding=1); s.out = torch.nn.Conv2d(C, 3, 3, padding=1)
        def forward(s, x, temb):
            h = s.r1(s.inp(x), temb); h = s.a(h) if p.get("attn", True) else h; h = s.r2(h, temb); return s.out(torch.nn.functional.interpolate(s.down(h), scale_factor=2))
    m = Stage().to(dev); x = torch.randn(B, 3, Hs, Hs, device=dev); temb = torch.randn(B, 128, device=dev); noise = torch.randn_like(x)
    params = list(m.parameters())
    def step():
        loss = ((m(x, temb) - noise) ** 2).mean(); grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for q, g in zip(params, grads): q.sub_(1e-4 * g)
        return [loss]
    return step, params

def b_contrastive(p, dev):
    """two-tower contrastive (InfoNCE) training with in-batch negatives"""
    B, D, Hd = p["batch"], p["dim"], p["hidden"]
    ta = MLP(D, Hd, 2, "gelu").to(dev); tb = MLP(D, Hd, 2, "gelu").to(dev); xa = torch.randn(B, D, device=dev); xb = torch.randn(B, D, device=dev)
    params = list(ta.parameters()) + list(tb.parameters()); labels = torch.arange(B, device=dev)
    def step():
        a = torch.nn.functional.normalize(ta(xa), dim=-1); b = torch.nn.functional.normalize(tb(xb), dim=-1); sim = a @ b.t() / p.get("temp", 0.07)
        loss = (torch.nn.functional.cross_entropy(sim, labels) + torch.nn.functional.cross_entropy(sim.t(), labels)) / 2
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for q, g in zip(params, grads): q.sub_(1e-3 * g)
        return [loss]
    return step, params

BUILDERS.update(hf2=b_hf2, tv=b_tv, train_real=b_train_real, dlrm=b_dlrm, gnn=b_gnn, moe=b_moe, unet_block=b_unet_block, contrastive=b_contrastive)

def grid_real2():
    S = []
    dec = ["mistral", "qwen2", "phi3", "gemma2", "opt", "bloom", "falcon", "gpt_neox", "gptj"]
    for arch, train in itertools.product(dec, [False, True]):
        S.append(_spec("hf2", "real", arch=arch, layers=2, hidden=1024, heads=16, kv_heads=4 if arch in ("mistral", "qwen2", "phi3", "gemma2", "falcon") else 16, batch=2, seq=512, train=train))
    for arch in ("mistral", "qwen2"):
        S.append(_spec("hf2", "real", arch=arch, layers=1, hidden=4096, heads=32, kv_heads=8, inter=14336 if arch == "mistral" else 11008, batch=1, seq=1024, train=False, dtype="bf16"))
        S.append(_spec("hf2", "real", arch=arch, layers=1, hidden=2048, heads=16, kv_heads=4, batch=1, seq=2048, train=True, dtype="bf16"))
    for arch, train in itertools.product(["roberta", "distilbert", "albert", "electra", "deberta_v2", "xlm_roberta"], [False, True]):
        S.append(_spec("hf2", "real", arch=arch, layers=2, hidden=768, heads=12, batch=8, seq=128, train=train))
    for arch, train in itertools.product(["bart", "whisper"], [False, True]):
        S.append(_spec("hf2", "real", arch=arch, layers=2, hidden=512, heads=8, batch=2, seq=128, train=train))
    for train in (False, True):
        S.append(_spec("hf2", "real", arch="wav2vec2", layers=2, hidden=768, heads=12, batch=2, samples=16000, train=train))
        S.append(_spec("hf2", "real", arch="clip", layers=2, hidden=512, heads=8, batch=8, seq=32, img=224, patch=32, train=train))
    for arch, train in itertools.product(["dinov2", "convnext", "swin", "resnet", "mobilenet_v2", "segformer"], [False, True]):
        S.append(_spec("hf2", "real", arch=arch, layers=2, hidden=768 if arch in ("dinov2",) else 512, heads=12 if arch == "dinov2" else 16, batch=4, train=train))
    for name, train in itertools.product(["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0", "densenet121", "vgg11_bn", "squeezenet1_0", "shufflenet_v2_x1_0", "regnet_y_400mf", "vit_b_16", "swin_t", "convnext_tiny"], [False, True]):
        S.append(_spec("tv", "real", name=name, batch=4, img=224, train=train))
    for arch, opt_, ac, accum, ls, clip in [("gpt2", "adamw", False, 1, 0.0, 1.0), ("gpt2", "adamw", True, 1, 0.0, 1.0), ("gpt2", "adamw", False, 2, 0.1, 1.0), ("gpt2", "adamw", False, 4, 0.0, 0.0), ("gpt2", "sgd_momentum", False, 1, 0.0, 1.0), ("gpt2", "adagrad", False, 1, 0.0, 1.0),
                                              ("llama", "adamw", False, 1, 0.0, 1.0), ("llama", "adamw", True, 1, 0.0, 1.0), ("llama", "adamw", True, 2, 0.1, 1.0), ("llama", "adamw", False, 4, 0.0, 0.0)]:
        S.append(_spec("train_real", "real", arch=arch, layers=2, hidden=768 if arch == "gpt2" else 1024, heads=12 if arch == "gpt2" else 8, batch=2, seq=512, optimizer=opt_, autocast=ac, accum=accum, label_smoothing=ls, clip=clip))
    for nt, rows, lookups in [(8, 10000, 4), (26, 100000, 1), (8, 1000000, 16)]:
        S.append(_spec("dlrm", "real", n_tables=nt, rows=rows, dim=64, dense=13, lookups=lookups, batch=2048))
    for nodes, edges, L in [(10000, 100000, 2), (100000, 1000000, 2), (10000, 100000, 8)]:
        S.append(_spec("gnn", "real", nodes=nodes, edges=edges, dim=128, layers=L))
    for E, k, D in [(8, 2, 1024), (64, 2, 1024), (8, 1, 4096)]:
        S.append(_spec("moe", "real", batch=4, seq=256, dim=D, experts=E, topk=k, ffn=4 * D if D <= 1024 else 2 * D))
    for ch, img, attn in [(64, 32, True), (128, 64, True), (128, 64, False), (256, 32, True)]:
        S.append(_spec("unet_block", "real", ch=ch, img=img, batch=4, attn=attn))
    for B, D in [(256, 512), (4096, 1024)]:
        S.append(_spec("contrastive", "real", batch=B, dim=D, hidden=1024))
    return S

GRIDS.update(real2=grid_real2)
