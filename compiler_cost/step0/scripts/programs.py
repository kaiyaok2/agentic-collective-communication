"""programs.py — registry of graph-producing programs for the compiler-cost corpus.

A *spec* is a JSON-able dict: {"id": str, "family": str, "source": "synth|real|mutation|collective", "params": {...},
"parent": optional id, "mutation": optional str}. `build(spec, dev)` returns (step_fn, keep) where step_fn() runs one
iteration on device `dev` and returns a list of output tensors; `keep` is a list of tensors (parameters) that stay live.
Everything is random-initialised; no downloads.
"""
import math, random, itertools, torch

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
