#!/usr/bin/env python3
"""C-Q2, leak-free: four NON-LLM searches over a GENERIC program-edit space, starting from each developer
baseline, same simulator objective, same correctness oracle (sandbox OR real-tensor), equal evaluation budget.
No rule here encodes anything about which rewrite is good for collectives.

Edit operators (generic AST edits):
  E1 swap collective primitive      xm.all_gather <-> xm.all_reduce <-> xm.reduce_scatter (args patched to a legal form)
  E2 mutate a constant              dim / scale / shard_count / integer literals (+-1, 0, 1, world_size, ...)
  E3 delete a statement             (anything but return)
  E4 duplicate / move a statement   (reorder within a block)
  E5 insert a tensor op             wrap an existing name in .contiguous() / .view(-1) / .reshape(...) / .unsqueeze(0) / .squeeze(0)
  E6 wrap into torch.stack/cat      replace an operand `x` by torch.stack([x, x2]) or torch.cat([...]) of two live names (+ index back)
  E7 loop unroll / loop fuse        unroll a range loop body (constant trip) or merge two adjacent range loops with same iter
Methods: random walk, GA (mutation+crossover over edit sequences), beam over edit sequences, and
"rule-based" = exhaustive application of standard compiler passes only (DCE, CSE, LICM, loop fusion).
Run ON master: cd /home/ubuntu/OverlayCCL && PYTHONPATH=. nice -n 19 $VENV/bin/python /home/ubuntu/ap7/generic_search.py
"""
import ast, copy, itertools, json, os, random, sys, time, threading
import torch
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO); sys.path.insert(0, "/home/ubuntu/ap7")
from search.problems import get_problem
import search.problems_modext
from search.correctness_test import benchmark_xla_candidate_generic, test_xla_candidate_generic, _compare_outputs_generic
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts
import rewriter_search as RW   # reuse: is_coll, src, real_oracle (ThreadedXM), RING_KV_BASELINE, BASELINE

WORLD, NODES = 224, 7
TOPO = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                         num_devices_per_node=16, cores_per_device=2, device_adjacency=None,
                         efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
SEND = make_send_counts("moe", world=WORLD)
BUDGET = int(os.environ.get("BUDGET", "150"))
SCORER = os.environ.get("SCORER", "default")   # default | pipeline (static _HARDWARE_MEASUREMENTS table) | measured (TABLE=hw_probe json)
SIM_KW = {}
if SCORER == "pipeline":
    from search.agent_simulator_config import _HARDWARE_MEASUREMENTS as HM
    comp = HM.get("compilation_cost_us", {}) or {}; mc = HM.get("memcpy_throughput", {}) or {}
    SIM_KW = dict(op_costs=dict(HM.get("xla_op_overhead_us", {})), dispatch_overhead_us=100.0,
                  graph_launch_overhead_us=(HM.get("graph_launch_overhead_us", {}) or {}).get("per_mark_step_us", 0.0),
                  compilation_cost_samples=comp.get("samples", []), compilation_load_events_per_run=comp.get("load_events_per_run", 2),
                  compilation_amortize_steps=comp.get("typical_training_steps", 5000),
                  pipeline_amort_alpha1=0.30, pipeline_amort_alpha2=0.10, pipeline_amort_alpha3=0.02,
                  memcpy_bytes_per_us=float(mc.get("strided_gbps", 0.0)) * 1000.0, memcpy_seq_bytes_per_us=float(mc.get("sequential_gbps", 0.0)) * 1000.0)
elif SCORER == "v4":
    # the rebuttal's faithful simulator: correctness_test_v4 + measured probe table + real harness shapes (score_v4.KW/REAL)
    os.environ.setdefault("TABLE", "/home/ubuntu/hwprobe/hw_probe_measured.json"); os.environ.setdefault("RESET", "fusion")
    os.environ.setdefault("TABLES", "0"); os.environ.setdefault("MULTS", '{"ring_kv":2,"alltoallv":32,"uniform_a2a":384}')
    os.environ.setdefault("MANIFEST", "/home/ubuntu/ap7/manifest_min.json"); json.dump([], open("/home/ubuntu/ap7/manifest_min.json", "w"))
    import score_v4 as _SV
    SIM_KW = dict(_SV.KW); _V4 = _SV
elif SCORER == "measured":
    P = json.load(open(os.environ["TABLE"])); DT = P["derived_table"]; SP = P["sim_params"]
    SIM_KW = dict(op_costs=dict(DT["xla_op_overhead_us"]), dispatch_overhead_us=SP["dispatch_overhead_us"],
                  graph_launch_overhead_us=SP["graph_launch_overhead_us"], compilation_cost_samples=SP["compilation_cost_samples"],
                  compilation_load_events_per_run=2, compilation_amortize_steps=5000,
                  pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
                  memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"])
SEED = int(os.environ.get("SEED", "0"))
random.seed(SEED)
COLL = ["all_gather", "all_reduce", "reduce_scatter"]
DEBUG = os.environ.get("DEBUG", "0") == "1"
ONLY = [x for x in os.environ.get("PROBLEMS", "").split(",") if x]

def src(n): return ast.unparse(n)
def stmts(fn):
    """all (block, index) statement sites, excluding the final return"""
    out = []
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if isinstance(body, list):
            for i, st in enumerate(body):
                if not isinstance(st, ast.Return): out.append((parent, i))
    return out
def live_names(fn):
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    names |= {a.arg for a in fn.args.args}
    return sorted(names - {"xm", "torch"})
def coll_calls(fn): return [n for n in ast.walk(fn) if RW.is_coll(n)]

# ------------------------------------------------------------------ generic edits (each returns a new fn or None)
def e1_swap_primitive(fn):
    fn = copy.deepcopy(fn); calls = coll_calls(fn)
    if not calls: return None
    c = random.choice(calls); cur = c.func.attr; new = random.choice([p for p in COLL if p != cur])
    # operand = args[1] for all_reduce/reduce_scatter, args[0] for all_gather
    operand = c.args[1] if cur in ("all_reduce", "reduce_scatter") and len(c.args) > 1 else c.args[0]
    c.func.attr = new
    if new == "all_gather": c.args = [operand]; c.keywords = [ast.keyword("dim", ast.Constant(random.choice([0, -1])))]
    elif new == "all_reduce": c.args = [ast.parse("xm.REDUCE_SUM").body[0].value, operand]; c.keywords = []
    else: c.args = [ast.parse("xm.REDUCE_SUM").body[0].value, operand]; c.keywords = [ast.keyword("scale", ast.Constant(1.0)), ast.keyword("scatter_dim", ast.Constant(0)), ast.keyword("shard_count", ast.Name("world_size", ast.Load()))]
    return fn
def e2_mutate_constant(fn):
    fn = copy.deepcopy(fn); consts = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    if not consts: return None
    c = random.choice(consts)
    if isinstance(c.value, int): c.value = random.choice([c.value + 1, c.value - 1, 0, 1, -1, 2])
    else: c.value = random.choice([c.value * 2, c.value / 2, 1.0, 0.5])
    return fn
def e3_delete_stmt(fn):
    fn = copy.deepcopy(fn); sites = stmts(fn)
    if not sites: return None
    parent, i = random.choice(sites)
    if len(parent.body) <= 1: return None
    del parent.body[i]; return fn
def e4_move_stmt(fn):
    fn = copy.deepcopy(fn); sites = stmts(fn)
    parent, i = random.choice(sites)
    if len(parent.body) < 3: return None
    st = parent.body.pop(i); j = random.randrange(0, len(parent.body)); parent.body.insert(j, st); return fn
def e5_insert_tensor_op(fn):
    fn = copy.deepcopy(fn); names = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in ("xm", "torch", "range", "len")]
    if not names: return None
    n = random.choice(names); op = random.choice([".contiguous()", ".view(-1)", ".reshape(-1)", ".unsqueeze(0)", ".squeeze(0)", ".float()"])
    new = ast.parse(f"{n.id}{op}").body[0].value
    class Rep(ast.NodeTransformer):
        done = False
        def visit_Name(self, node):
            if node is n and not self.done: self.done = True; return new
            return node
    return Rep().visit(fn)
def e6_stack_operands(fn):
    fn = copy.deepcopy(fn); calls = coll_calls(fn)
    if not calls: return None
    c = random.choice(calls); k = 1 if c.func.attr in ("all_reduce", "reduce_scatter") else 0
    if len(c.args) <= k: return None
    live = live_names(fn)
    if len(live) < 2: return None
    a, b = random.sample(live, 2); how = random.choice(["stack", "cat"])
    c.args[k] = ast.parse(f"torch.{how}([{a}, {b}], dim=0)").body[0].value
    return fn
def e7_loop_transform(fn):
    fn = copy.deepcopy(fn); loops = [(p, i) for p, i in stmts(fn) if isinstance(p.body[i], ast.For)]
    if not loops: return None
    p, i = random.choice(loops); loop = p.body[i]
    if random.random() < 0.5 and i + 1 < len(p.body) and isinstance(p.body[i + 1], ast.For) and src(p.body[i + 1].iter) == src(loop.iter):
        nxt = p.body[i + 1]  # loop fusion
        if isinstance(nxt.target, ast.Name) and isinstance(loop.target, ast.Name):
            ren = nxt.target.id
            class R(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == ren: node.id = loop.target.id
                    return node
            fused_body = [R().visit(copy.deepcopy(s)) for s in nxt.body]
            loop.body = loop.body + fused_body; del p.body[i + 1]; return fn
    # unroll by 2 (generic): for v in range(N): body  ->  for v in range(0, N, 2): body; body[v+1]
    if isinstance(loop.iter, ast.Call) and getattr(loop.iter.func, "id", "") == "range" and len(loop.iter.args) == 1 and isinstance(loop.target, ast.Name):
        v = loop.target.id
        class Shift(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == v and isinstance(node.ctx, ast.Load): return ast.BinOp(left=ast.Name(v, ast.Load()), op=ast.Add(), right=ast.Constant(1))
                return node
        second = [Shift().visit(copy.deepcopy(s)) for s in loop.body]
        loop.iter = ast.parse(f"range(0, {src(loop.iter.args[0])}, 2)").body[0].value
        loop.body = loop.body + second
        return fn
    return None
EDITS = [e1_swap_primitive, e2_mutate_constant, e3_delete_stmt, e4_move_stmt, e5_insert_tensor_op, e6_stack_operands, e7_loop_transform]

# ------------------------------------------------------------------ standard compiler passes (rule-based arm)
def pass_dce(fn):
    """remove assignments to names never read (iterate to fixpoint)"""
    fn = copy.deepcopy(fn); changed = True
    while changed:
        changed = False
        reads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for parent, i in stmts(fn):
            st = parent.body[i]
            if isinstance(st, ast.Assign) and all(isinstance(t, ast.Name) and t.id not in reads for t in st.targets) and not any(RW.is_coll(n) for n in ast.walk(st)):
                del parent.body[i]; changed = True; break
    return fn
def pass_cse(fn):
    """common-subexpression elimination over identical RHS assignments in the same block"""
    fn = copy.deepcopy(fn)
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list): continue
        seen = {}
        for st in list(body):
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                key = src(st.value)
                if key in seen:
                    st.value = ast.Name(seen[key], ast.Load())
                else: seen[key] = st.targets[0].id
    return fn
def pass_licm(fn):
    """loop-invariant code motion: hoist assignments whose RHS does not mention the loop variable or loop-defined names"""
    fn = copy.deepcopy(fn)
    for parent, i in stmts(fn):
        st = parent.body[i]
        if isinstance(st, ast.For) and isinstance(st.target, ast.Name):
            defined = {n.id for b in st.body for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | {st.target.id}
            hoist = [b for b in st.body if isinstance(b, ast.Assign) and not ({n.id for n in ast.walk(b.value) if isinstance(n, ast.Name)} & defined)]
            for b in hoist: st.body.remove(b)
            parent.body[i:i] = hoist
            return fn
    return fn
def pass_fuse(fn):
    """fuse adjacent loops with identical iteration space"""
    fn = copy.deepcopy(fn)
    for parent, i in stmts(fn):
        if i + 1 < len(parent.body) and isinstance(parent.body[i], ast.For) and isinstance(parent.body[i + 1], ast.For) \
           and src(parent.body[i].iter) == src(parent.body[i + 1].iter) and isinstance(parent.body[i].target, ast.Name) and isinstance(parent.body[i + 1].target, ast.Name):
            a, b = parent.body[i], parent.body[i + 1]; ren, to = b.target.id, a.target.id
            class R(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == ren: node.id = to
                    return node
            a.body += [R().visit(copy.deepcopy(s)) for s in b.body]; del parent.body[i + 1]; return fn
    return fn
PASSES = {"DCE": pass_dce, "CSE": pass_cse, "LICM": pass_licm, "LoopFusion": pass_fuse}

# ------------------------------------------------------------------ objective + oracle
_cache = {}
def evaluate(problem, code):
    """-> (ok, sim_us). Cached by source."""
    if code in _cache: return _cache[code]
    try:
        ns = {}; exec(code, ns); fn = ns[problem.evolved_fn_name]; fn.__candidate_source__ = code
    except Exception:
        _cache[code] = (False, None); return _cache[code]
    ok = False
    try:
        r = test_xla_candidate_generic(problem, fn, world_sizes=[8], verbose=False); ok = bool(r[0]) if isinstance(r, tuple) else bool(r)
    except Exception: ok = False
    if not ok:
        try: ok = bool(RW.real_oracle(problem, fn)[0])
        except Exception: ok = False
    v = None
    if ok:
        try:
            kw = dict(SIM_KW)
            if SCORER == "v4":
                kw["training_scale_bytes_multiplier"] = _V4.MULTS.get(problem.name, _V4.DTYPE_MULT.get(problem.name, 1.0) if problem.name in _V4.MULT1 else 1.0)
                b = _V4.ct.benchmark_xla_candidate_generic(problem, fn, TOPO, SEND, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **kw)
            else:
                if kw:
                    from experiments.run_search import _problem_train_scale_multiplier
                    kw["training_scale_bytes_multiplier"] = _problem_train_scale_multiplier(problem)
                b = benchmark_xla_candidate_generic(problem, fn, TOPO, SEND, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **kw)
            v = None if "error" in b else b["sim_time_us"]
        except Exception: v = None
    _cache[code] = (ok and v is not None, v); return _cache[code]

def mutate(fn_ast, n_edits=1):
    a = fn_ast
    for _ in range(n_edits):
        for _try in range(6):
            e = random.choice(EDITS)
            try: b = e(a)
            except Exception: b = None
            if b is not None:
                try: ast.fix_missing_locations(b); src(b); a = b; break
                except Exception: continue
    return a

def run_random(problem, start, budget):
    best = (evaluate(problem, src(start))[1], src(start)); evals = 1
    while evals < budget:
        cand = mutate(start, random.randint(1, 3)); ok, v = evaluate(problem, src(cand)); evals += 1
        if ok and v < best[0]: best = (v, src(cand))
    return dict(best_sim=best[0], code=best[1], evals=evals)

def run_ga(problem, start, budget, pop=12):
    s0 = evaluate(problem, src(start))[1]; evals = 1
    population = [(s0, start)] + [(None, mutate(start, random.randint(1, 3))) for _ in range(pop - 1)]
    scored = []
    for v, a in population:
        if v is None: ok, v = evaluate(problem, src(a)); evals += 1; v = v if ok else float("inf")
        scored.append((v, a))
    while evals < budget:
        scored.sort(key=lambda t: t[0]); parents = scored[:max(2, pop // 3)]
        pa, pb = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
        child = copy.deepcopy(pa[1])
        # crossover: replace a random statement block of child with the corresponding block from pb (by position)
        sa, sb = stmts(child), stmts(pb[1])
        if sa and sb:
            (p1, i1), (p2, i2) = random.choice(sa), random.choice(sb)
            p1.body[i1] = copy.deepcopy(p2.body[i2])
        child = mutate(child, 1); ok, v = evaluate(problem, src(child)); evals += 1
        scored.append((v if ok else float("inf"), child)); scored.sort(key=lambda t: t[0]); scored = scored[:pop]
    best = scored[0]; return dict(best_sim=None if best[0] == float("inf") else best[0], code=src(best[1]), evals=evals)

def run_beam(problem, start, budget, width=4, depth=5, branch=8):
    s0 = evaluate(problem, src(start))[1]; evals = 1; frontier = [(s0, start)]; best = (s0, src(start)); seen = {src(start)}
    for _ in range(depth):
        cands = []
        for v, a in frontier:
            for _ in range(branch):
                c = mutate(a, 1); code = src(c)
                if code in seen: continue
                seen.add(code); ok, cv = evaluate(problem, code); evals += 1
                if ok: cands.append((cv, c))
                if evals >= budget: break
            if evals >= budget: break
        if not cands: break
        cands.sort(key=lambda t: t[0]); frontier = cands[:width]
        if cands[0][0] < best[0]: best = (cands[0][0], src(cands[0][1]))
        if evals >= budget: break
    return dict(best_sim=best[0], code=best[1], evals=evals)

def run_rules(problem, start):
    """apply standard compiler passes to a fixpoint (and all orderings of up to 3 passes), keep the best correct program"""
    s0 = evaluate(problem, src(start))[1]; best = (s0, src(start), []); evals = 1
    for seq in itertools.chain.from_iterable(itertools.permutations(PASSES, k) for k in (1, 2, 3)):
        a = start
        for name in seq:
            try: a = PASSES[name](a); ast.fix_missing_locations(a)
            except Exception: a = None; break
        if a is None: continue
        code = src(a); ok, v = evaluate(problem, code); evals += 1
        if ok and v < best[0]: best = (v, code, list(seq))
    return dict(best_sim=best[0], code=best[1], passes=best[2], evals=evals)

def structure(code):
    t = ast.parse(code); c = {}
    for n in ast.walk(t):
        if RW.is_coll(n): c[n.func.attr] = c.get(n.func.attr, 0) + 1
    return dict(colls=c, loops=sum(isinstance(n, ast.For) for n in ast.walk(t)))

if __name__ == "__main__":
    out = {}
    for pname, bname in RW.BASELINE.items():
        if ONLY and pname not in ONLY: continue
        problem = get_problem(pname); code = RW.RING_KV_BASELINE if pname == "ring_kv" else problem.builtin_templates[bname]
        if SCORER == "v4" and pname in _V4.REAL: _V4.REAL[pname](problem)   # trace at the real harness shapes
        start = ast.parse(code).body[0]; t0 = time.time(); _cache.clear()
        res = {}
        for name, fnc in [("random", lambda: run_random(problem, start, BUDGET)), ("ga", lambda: run_ga(problem, start, BUDGET)),
                          ("beam", lambda: run_beam(problem, start, BUDGET)), ("rules", lambda: run_rules(problem, start))]:
            try: r = fnc()
            except Exception as e: r = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            if "code" in r: r["structure"] = structure(r["code"])
            res[name] = r
        res["start_sim"] = evaluate(problem, code)[1]; res["start_structure"] = structure(code); res["secs"] = round(time.time() - t0, 1)
        out[pname] = res
        print(f"{pname:15s} start={res['start_sim']} | " + " | ".join(f"{k}={res[k].get('best_sim')} {res[k].get('structure',{}).get('colls')}" for k in ("random", "ga", "beam", "rules")) + f" | {res['secs']}s", flush=True)
    os.makedirs("/home/ubuntu/sweep_results", exist_ok=True)
    _od = os.environ.get("OUTDIR", "/home/ubuntu/sweep_results"); os.makedirs(_od, exist_ok=True); json.dump(out, open(f"{_od}/generic_search_{SCORER}_s{SEED}.json", "w"), indent=1)
    print("wrote generic_search json")
