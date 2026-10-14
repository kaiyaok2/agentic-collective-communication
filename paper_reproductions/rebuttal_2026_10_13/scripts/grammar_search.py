#!/usr/bin/env python3
"""Non-LLM Phase-3 baselines for C-Q2: beam search and a genetic algorithm over a hand-built
STRUCTURAL GRAMMAR of strategies, scored by the same simulator (benchmark_xla_candidate_generic,
default calibration) at world=224 / 7 nodes, under a fixed evaluation budget.

The grammar is the honest part: it hard-codes the move set (bundle across microbatches / layers,
stack vs cat vs flatten, AG+RS vs AG+slice, 1-AG vs 3-AR vs MAX+stacked-SUM). Beam/GA can only find
what the grammar can express; the LLM's role in the paper is to supply such moves.

Run ON master:  cd /home/ubuntu/OverlayCCL && PYTHONPATH=. $VENV/bin/python /home/ubuntu/ap7/grammar_search.py
Writes /home/ubuntu/sweep_results/grammar_search.json
"""
import itertools, json, os, random, sys, time
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
WORLD, NODES = 224, 7
BUDGET = int(os.environ.get("BUDGET", "40"))   # simulator evaluations per problem per method
random.seed(0)
_sim = {}
def _init_sim():
    if _sim: return
    from search.problems import get_problem
    from search.correctness_test import benchmark_xla_candidate_generic
    from simulator.topology import MultiNodeTopology
    from experiments.run_search import make_send_counts
    _sim["get_problem"] = get_problem; _sim["bench"] = benchmark_xla_candidate_generic
    _sim["topo"] = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                                     num_devices_per_node=16, cores_per_device=2, device_adjacency=None)
    _sim["send"] = make_send_counts("moe", world=WORLD)

# ----------------------------------------------------------------------------- grammars
# Each grammar: dict of dimension -> list of options; render(genome) -> code string for evolved_fn.
G = {}

def _pp(g):
    bundle, pack = g["bundle"], g["pack"]
    if bundle == "per_mb":
        return '''def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    outs = []
    for m in range(M):
        buf = torch.zeros(half, *activations[0].shape, dtype=activations[0].dtype, device=activations[0].device)
        if stage == src_stage:
            buf[pair_id] = activations[m]
        outs.append(xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id])
    return outs
'''
    body_pack = {"stack": "stacked = torch.stack(activations, dim=0)",
                 "cat": "stacked = torch.cat([a.unsqueeze(0) for a in activations], dim=0)"}[pack]
    return f'''def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    buf = torch.zeros(half, M, *a0.shape, dtype=a0.dtype, device=a0.device)
    if stage == src_stage:
        {body_pack}
        buf[pair_id] = stacked
    ared = xm.all_reduce(xm.REDUCE_SUM, buf)
    return [ared[pair_id, m] for m in range(M)]
'''
G["pp_send_recv"] = dict(dims={"bundle": ["per_mb", "all_mb"], "pack": ["stack", "cat"]}, render=_pp)

def _tp(g):
    b, pack = g["bundle"], g["pack"]
    if b == "per_call":
        return '''def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [[xm.all_reduce(xm.REDUCE_SUM, partials[m][L]) for L in range(N_LAYERS)] for m in range(M)]
'''
    if b == "per_mb":
        return '''def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(partials[m], dim=0)
        r = xm.all_reduce(xm.REDUCE_SUM, st)
        outs.append([r[L] for L in range(N_LAYERS)])
    return outs
'''
    if b == "per_layer":
        return '''def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = [[None] * N_LAYERS for _ in range(M)]
    for L in range(N_LAYERS):
        st = torch.stack([partials[m][L] for m in range(M)], dim=0)
        r = xm.all_reduce(xm.REDUCE_SUM, st)
        for m in range(M):
            outs[m][L] = r[m]
    return outs
'''
    pk = {"stack": "st = torch.stack([partials[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)\n    r = xm.all_reduce(xm.REDUCE_SUM, st)\n    return [[r[m * N_LAYERS + L] for L in range(N_LAYERS)] for m in range(M)]",
          "flatten": "shapes = [partials[m][L].shape for m in range(M) for L in range(N_LAYERS)]\n    flat = torch.cat([partials[m][L].flatten() for m in range(M) for L in range(N_LAYERS)], dim=0)\n    r = xm.all_reduce(xm.REDUCE_SUM, flat)\n    outs = []; off = 0\n    for m in range(M):\n        row = []\n        for L in range(N_LAYERS):\n            n = 1\n            for d in shapes[m * N_LAYERS + L]: n *= d\n            row.append(r[off:off + n].reshape(shapes[m * N_LAYERS + L])); off += n\n        outs.append(row)\n    return outs"}[pack]
    return f'''def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    {pk}
'''
G["tp_mlp"] = dict(dims={"bundle": ["per_call", "per_mb", "per_layer", "all"], "pack": ["stack", "flatten"]}, render=_tp)

def _fsdp(g):
    b = g["bundle"]
    if b == "per_call":
        return '''def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [[xm.all_gather(shards[m][L], dim=-1) for L in range(N_LAYERS)] for m in range(M)]
'''
    if b == "per_mb":
        return '''def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        g = xm.all_gather(st, dim=-1)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
'''
    return '''def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([shards[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
    g = xm.all_gather(st, dim=-1)
    return [[g[m * N_LAYERS + L] for L in range(N_LAYERS)] for m in range(M)]
'''
G["fsdp_prefetch"] = dict(dims={"bundle": ["per_call", "per_mb", "all"]}, render=_fsdp)

def _lb(g):
    if g["bundle"] == "two_ar":
        return '''def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_reduce(xm.REDUCE_SUM, attn_partial), xm.all_reduce(xm.REDUCE_SUM, mlp_partial)]
'''
    pk = {"stack": "st = torch.stack([attn_partial, mlp_partial], dim=0)\n    r = xm.all_reduce(xm.REDUCE_SUM, st)\n    return [r[0], r[1]]",
          "cat": "n = attn_partial.numel()\n    flat = torch.cat([attn_partial.flatten(), mlp_partial.flatten()], dim=0)\n    r = xm.all_reduce(xm.REDUCE_SUM, flat)\n    return [r[:n].reshape(attn_partial.shape), r[n:].reshape(mlp_partial.shape)]"}[g["pack"]]
    return f'''def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    {pk}
'''
G["llama_block_ar"] = dict(dims={"bundle": ["two_ar", "one_ar"], "pack": ["stack", "cat"]}, render=_lb)

def _dxe(g):
    s = g["structure"]
    if s == "one_ag":
        return '''def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    full = xm.all_gather(logits_local, dim=1)
    m = full.max(dim=-1).values
    lse = (full - m.unsqueeze(-1)).exp().sum(dim=-1).log() + m
    tgt = full.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (lse - tgt).mean()
'''
    if s == "three_ar":
        return '''def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    lo, hi = rank * V_local, (rank + 1) * V_local
    local_max = logits_local.max(dim=-1).values
    gmax = xm.all_reduce(xm.REDUCE_MAX, local_max)
    sumexp = (logits_local - gmax.unsqueeze(-1)).exp().sum(dim=-1)
    gsum = xm.all_reduce(xm.REDUCE_SUM, sumexp)
    in_shard = (targets >= lo) & (targets < hi)
    t = logits_local.gather(1, (targets - lo).clamp(0, V_local - 1).unsqueeze(1)).squeeze(1)
    t = torch.where(in_shard, t, torch.zeros_like(t))
    gt = xm.all_reduce(xm.REDUCE_SUM, t)
    return (gsum.log() + gmax - gt).mean()
'''
    return '''def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    lo, hi = rank * V_local, (rank + 1) * V_local
    local_max = logits_local.max(dim=-1).values
    gmax = xm.all_reduce(xm.REDUCE_MAX, local_max)
    sumexp = (logits_local - gmax.unsqueeze(-1)).exp().sum(dim=-1)
    in_shard = (targets >= lo) & (targets < hi)
    t = logits_local.gather(1, (targets - lo).clamp(0, V_local - 1).unsqueeze(1)).squeeze(1)
    t = torch.where(in_shard, t, torch.zeros_like(t))
    pack = torch.stack([sumexp, t], dim=1)
    g = xm.all_reduce(xm.REDUCE_SUM, pack)
    return (g[:, 0].log() + gmax - g[:, 1]).mean()
'''
G["dxe"] = dict(dims={"structure": ["one_ag", "three_ar", "max_plus_stacked_sum"]}, render=_dxe)

def _a2av(g):
    s, ext = g["structure"], g["extract"]
    pack = '''    pack_size = world_size * max_chunk
    packed = torch.zeros(pack_size, device=input_tensor.device, dtype=input_tensor.dtype)
    off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = input_tensor[off:off + sc]
        off += sc
'''
    if s == "ag_rs":
        return f'''def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
{pack}    gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
    transposed = gathered.view(world_size, world_size, max_chunk).permute(1, 0, 2).contiguous().view(-1)
    mine = xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0 / world_size, scatter_dim=0, shard_count=world_size)
    return torch.cat([mine[s * max_chunk:s * max_chunk + recv_counts[s]] for s in range(world_size)])
'''
    ex = {"loop_cat": "    chunks = []\n    for src in range(world_size):\n        base = src * pack_size + rank * max_chunk\n        chunks.append(gathered[base:base + recv_counts[src]])\n    return torch.cat(chunks)",
          "view_col": "    col = gathered.view(world_size, world_size, max_chunk)[:, rank, :].contiguous()\n    return torch.cat([col[s, :recv_counts[s]] for s in range(world_size)])"}[ext]
    return f'''def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
{pack}    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(-1)
{ex}
'''
G["alltoallv"] = dict(dims={"structure": ["ag_rs", "ag_slice"], "extract": ["loop_cat", "view_col"]}, render=_a2av)

def _ua2a(g):
    if g["structure"] == "ag_rs":
        return '''def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    transposed = gathered.view(world_size, world_size, chunk_size).permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0 / world_size, scatter_dim=0, shard_count=world_size)
'''
    ex = {"loop_cat": "    flat = gathered.view(world_size, -1)\n    return torch.cat([flat[src, rank * chunk_size:(rank + 1) * chunk_size] for src in range(world_size)], dim=0)",
          "view_col": "    return gathered.view(world_size, world_size, chunk_size)[:, rank, :].contiguous().view(-1)"}[g["extract"]]
    return f'''def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
{ex}
'''
G["uniform_a2a"] = dict(dims={"structure": ["ag_rs", "ag_slice"], "extract": ["loop_cat", "view_col"]}, render=_ua2a)

def _rkv(g):
    if g["bundle"] == "per_slot":
        return '''def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_gather(c.unsqueeze(0), dim=0).view(-1) for c in kv_chunks]
'''
    if g["bundle"] == "per_chunk":
        return '''def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for c in kv_chunks:
        n = c.numel() // 8
        parts = [xm.all_gather(c[i * n:(i + 1) * n].unsqueeze(0), dim=0) for i in range(8)]
        outs.append(torch.cat([p.view(world_size, -1) for p in parts], dim=1).reshape(-1))
    return outs
'''
    return '''def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    sizes = [c.numel() for c in kv_chunks]
    flat = torch.cat(kv_chunks, dim=0)
    g = xm.all_gather(flat.unsqueeze(0), dim=0)
    outs = []; off = 0
    for sz in sizes:
        outs.append(g[:, off:off + sz].reshape(-1)); off += sz
    return outs
'''
G["ring_kv"] = dict(dims={"bundle": ["per_chunk", "per_slot", "joint"]}, render=_rkv)

# ----------------------------------------------------------------------------- scoring
_cache = {}
def score(problem, pname, genome):
    key = (pname, tuple(sorted(genome.items())))
    if key in _cache: return _cache[key]
    code = G[pname]["render"](genome)
    try:
        ns = {}; exec(code, ns); fn = ns[problem.evolved_fn_name]
        fn.__candidate_source__ = code
        _init_sim()
        b = _sim["bench"](problem, fn, _sim["topo"], _sim["send"], WORLD, num_nodes=NODES,
                          unsupported_primitives=["all_to_all"])
        v = None if "error" in b else b["sim_time_us"]
    except Exception as e:
        v = None
    _cache[key] = v
    return v

def neighbors(pname, genome):
    out = []
    for d, opts in G[pname]["dims"].items():
        for o in opts:
            if o != genome[d]:
                n = dict(genome); n[d] = o; out.append(n)
    return out

def all_genomes(pname):
    dims = G[pname]["dims"]; keys = list(dims)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[dims[k] for k in keys])]

def run_beam(problem, pname, start, width=2, budget=BUDGET):
    evals = 0; beam = [start]; best = (score(problem, pname, start), start); evals += 1
    seen = {tuple(sorted(start.items()))}
    while evals < budget:
        cands = []
        for g in beam:
            for n in neighbors(pname, g):
                k = tuple(sorted(n.items()))
                if k in seen: continue
                seen.add(k); v = score(problem, pname, n); evals += 1
                if v is not None: cands.append((v, n))
                if evals >= budget: break
            if evals >= budget: break
        if not cands: break
        cands.sort(key=lambda t: t[0]); beam = [g for _, g in cands[:width]]
        if cands[0][0] is not None and (best[0] is None or cands[0][0] < best[0]): best = cands[0]
    return best, evals

def run_ga(problem, pname, start, pop=4, budget=BUDGET):
    evals = 0; population = [start] + [random.choice(all_genomes(pname)) for _ in range(pop - 1)]
    scored = []
    for g in population:
        v = score(problem, pname, g); evals += 1; scored.append((v if v is not None else float("inf"), g))
    while evals < budget:
        scored.sort(key=lambda t: t[0]); parents = [g for _, g in scored[:2]]
        child = {d: random.choice([parents[0][d], parents[1][d]]) for d in G[pname]["dims"]}
        d = random.choice(list(G[pname]["dims"])); child[d] = random.choice(G[pname]["dims"][d])   # mutation
        v = score(problem, pname, child); evals += 1
        scored.append((v if v is not None else float("inf"), child)); scored.sort(key=lambda t: t[0]); scored = scored[:pop]
    best = scored[0]; return (None if best[0] == float("inf") else best[0], best[1]), evals

def run_random(problem, pname, budget=BUDGET):
    gs = all_genomes(pname); best = (None, None); evals = 0
    for _ in range(budget):
        g = random.choice(gs); v = score(problem, pname, g); evals += 1
        if v is not None and (best[0] is None or v < best[0]): best = (v, g)
    return best, evals

START = {"pp_send_recv": {"bundle": "per_mb", "pack": "stack"}, "tp_mlp": {"bundle": "per_call", "pack": "stack"},
         "fsdp_prefetch": {"bundle": "per_call"}, "llama_block_ar": {"bundle": "two_ar", "pack": "stack"},
         "dxe": {"structure": "one_ag"}, "alltoallv": {"structure": "ag_rs", "extract": "loop_cat"},
         "uniform_a2a": {"structure": "ag_rs", "extract": "loop_cat"}, "ring_kv": {"bundle": "per_chunk"}}

if __name__ == "__main__":
    results = {}
    for pname in G:
        try:
            _init_sim(); problem = _sim["get_problem"](pname)
        except Exception as e:
            print(pname, "get_problem failed:", e); continue
        t0 = time.time()
        exhaustive = {json.dumps(g, sort_keys=True): score(problem, pname, g) for g in all_genomes(pname)}
        valid = {k: v for k, v in exhaustive.items() if v is not None}
        opt = min(valid.items(), key=lambda kv: kv[1]) if valid else (None, None)
        beam, eb = run_beam(problem, pname, START[pname])
        ga, eg = run_ga(problem, pname, START[pname])
        rnd, er = run_random(problem, pname)
        results[pname] = dict(grammar_size=len(exhaustive), valid=len(valid), exhaustive_opt=opt,
                              beam=dict(best=beam[0], genome=beam[1], evals=eb), ga=dict(best=ga[0], genome=ga[1], evals=eg),
                              random=dict(best=rnd[0], genome=rnd[1], evals=er), baseline=dict(genome=START[pname], sim=exhaustive[json.dumps(START[pname], sort_keys=True)]),
                              all=exhaustive, secs=round(time.time() - t0, 1))
        print(f"{pname:15s} grammar={len(exhaustive)} valid={len(valid)} opt={opt} | beam={beam[0]} {beam[1]} ({eb} evals) | ga={ga[0]} ({eg}) | random={rnd[0]} ({er}) | baseline={results[pname]['baseline']['sim']}", flush=True)
    os.makedirs("/home/ubuntu/sweep_results", exist_ok=True)
    json.dump(results, open("/home/ubuntu/sweep_results/grammar_search.json", "w"), indent=1)
    print("wrote grammar_search.json")
