#!/usr/bin/env python3
"""C-Q2: a NON-LLM generator. Starting from each developer baseline program, apply generic
source-level rewrite rules and search (beam / random walk) with the simulator as the objective and the
pipeline's correctness oracle as the gate. Reports whether the search converges to OverlayCCL's
deployed strategy structure, in how many simulator evaluations, and which rules were needed.

Rules (generic, problem-agnostic patterns over the AST):
  R1 hoist_loop_collective : a `for` loop whose body issues one collective on a loop-dependent operand
                             -> stack the operands, issue ONE collective, index the result
  R2 merge_adjacent        : two consecutive same-op collectives on independent tensors -> one on stack
  R3 ag_rs_to_ag_slice     : all_gather -> view/permute(1,0,2)/contiguous -> reduce_scatter(SUM,1/ws)
                             -> all_gather -> take own column (routing identity)
  R4 reduce_through_gather : row-reduction (max / exp-sum / one-hot gather) of an all_gather(dim=1)
                             -> local reduction + all_reduce (log-sum-exp decomposition)
Run ON master: cd /home/ubuntu/OverlayCCL && PYTHONPATH=. $VENV/bin/python /home/ubuntu/ap7/rewriter_search.py
"""
import ast, copy, itertools, json, os, random, sys, time, re
import torch
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
from search.problems import get_problem, PROBLEMS
import search.problems_modext  # registers Llama-side problems
from search.correctness_test import benchmark_xla_candidate_generic, test_xla_candidate_generic
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts

WORLD, NODES = 224, 7
TOPO = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5,
                         num_devices_per_node=16, cores_per_device=2, device_adjacency=None)
SEND = make_send_counts("moe", world=WORLD)
BUDGET = int(os.environ.get("BUDGET", "40"))
DEBUG = os.environ.get("DEBUG", "0") == "1"
ONLY = [x for x in os.environ.get("PROBLEMS", "").split(",") if x]
random.seed(0)
COLL = {"all_reduce", "all_gather", "reduce_scatter", "collective_permute"}
BASELINE = {"alltoallv": "allgather_reduce_scatter", "uniform_a2a": "allgather_reduce_scatter",
            "ring_kv": "per_chunk_per_part_allgather", "dxe": "full_vocab_ag", "pp_send_recv": "per_mb",
            "tp_mlp": "per_mb", "fsdp_prefetch": "per_mb", "llama_block_ar": "sequential_2ar"}


# The registered ring_kv "per_chunk_per_part_allgather" template reassembles parts in part-major order and
# therefore fails the problem's own reference (we verified with both oracles); the developer baseline in the
# training harness is per-head all_gather with rank-major reassembly. Start from that corrected form.
RING_KV_BASELINE = '''def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    N_PARTS = 16
    results = []
    for chunk in kv_chunks:
        part_sz = chunk.numel() // N_PARTS
        parts = []
        for p in range(N_PARTS):
            piece = chunk[p * part_sz:(p + 1) * part_sz]
            gathered = xm.all_gather(piece.unsqueeze(0), dim=0)
            parts.append(gathered.view(world_size, -1))
        results.append(torch.cat(parts, dim=1).reshape(-1))
    return results
'''

def is_coll(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and node.func.value.id == "xm" and node.func.attr in COLL)
def names_in(node): return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
def src(node): return ast.unparse(node)
def P(s): return ast.parse(s).body[0] if "\n" not in s.strip() else ast.parse(s).body

# ---------------------------------------------------------------- R1: hoist a collective out of a loop
_uid = [0]
def _normalize_appends(loop):
    """x.append(xm.coll(...)[...]) -> _cN = xm.coll(...); x.append(_cN[...])  (so the collective is an Assign)"""
    newbody = []
    for b in loop.body:
        if isinstance(b, ast.Expr) and isinstance(b.value, ast.Call) and getattr(b.value.func, "attr", "") == "append" and b.value.args:
            arg = b.value.args[0]
            colls = [n for n in ast.walk(arg) if is_coll(n)]
            if len(colls) == 1:
                _uid[0] += 1; tmp = f"_c{_uid[0]}"
                c = colls[0]
                newbody.append(ast.Assign(targets=[ast.Name(tmp, ast.Store())], value=copy.deepcopy(c)))
                class Rep(ast.NodeTransformer):
                    def visit_Call(self, node):
                        if node is c: return ast.Name(tmp, ast.Load())
                        return self.generic_visit(node)
                b = Rep().visit(b)
        newbody.append(b)
    loop.body = newbody

def r1_sites(fn):
    out = []
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list): continue
        for i, st in enumerate(body):
            if isinstance(st, ast.For) and isinstance(st.target, ast.Name) and isinstance(st.iter, ast.Call) \
               and getattr(st.iter.func, "id", "") == "range":
                tmp = copy.deepcopy(st); _normalize_appends(tmp)
                colls = [b for b in tmp.body if isinstance(b, ast.Assign) and is_coll(b.value)]
                nested = [b for b in tmp.body if isinstance(b, (ast.For, ast.While)) and any(is_coll(n) for n in ast.walk(b))]
                if len(colls) == 1 and not nested:
                    out.append((parent, i))
    return out

def r1_apply(fn, site):
    fn = copy.deepcopy(fn); parent, i = site
    plist = [p for p in ast.walk(fn) if isinstance(getattr(p, "body", None), list)]
    cand = [p for p in plist if i < len(p.body) and isinstance(p.body[i], ast.For) and src(p.body[i]) == src(parent.body[i])]
    if not cand: return None
    parent = cand[0]; loop = parent.body[i]; _normalize_appends(loop); v = loop.target.id
    js = [j for j, b in enumerate(loop.body) if isinstance(b, ast.Assign) and is_coll(b.value)]
    if len(js) != 1: return None
    j = js[0]; cst = loop.body[j]; call = cst.value; op = call.func.attr
    k = 1 if op in ("all_reduce", "reduce_scatter") else 0
    operand = call.args[k]
    pre_all = loop.body[:j]; post = loop.body[j + 1:]
    # backward slice: keep in `pre` only statements the operand (transitively) depends on; the rest moves to post
    need = set(names_in(operand)); pre = []
    for b in reversed(pre_all):
        defs = {n.id for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        defs |= {n.func.value.id for n in ast.walk(b) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)}
        if isinstance(b, ast.If) or (defs & need):
            pre.insert(0, b); need |= names_in(b)
        else:
            post.insert(0, b)
    dep = names_in(operand) | (set().union(*[names_in(b) for b in pre]) if pre else set())
    if v not in dep: return None
    if any(is_coll(n) for b in pre for n in ast.walk(b)): return None
    res_name = cst.targets[0].id
    _uid[0] += 1; u = _uid[0]
    build = ast.For(target=ast.Name(v, ast.Store()), iter=copy.deepcopy(loop.iter), orelse=[],
                    body=copy.deepcopy(pre) + [P(f"_ops{u}.append({src(operand)})")])
    stacked_call = copy.deepcopy(call); stacked_call.args[k] = ast.Name(f"_st{u}", ast.Load())
    stack_dim, index_expr = 0, f"_R{u}[{v}]"
    if op == "all_gather":
        gdim = None
        for kw in stacked_call.keywords:
            if kw.arg == "dim" and isinstance(kw.value, ast.Constant): gdim = kw.value.value
        if gdim is None and len(stacked_call.args) > 1 and isinstance(stacked_call.args[1], ast.Constant): gdim = stacked_call.args[1].value
        if gdim is not None and gdim == 0:
            # concatenate the operands along the gather dim; the gathered result is rank-major over (rank, piece)
            index_expr = f"_R{u}.reshape(world_size, len(_ops{u}), -1)[:, {v}].contiguous().reshape(world_size, *_ops{u}[{v}].shape[1:])"
            cat_mode = True
        elif gdim is not None and gdim > 0:
            stack_dim = gdim + 1; index_expr = f"_R{u}[" + ", ".join([":"] * (gdim + 1)) + f", {v}].contiguous()"
    stack_stmt = P(f"_st{u} = torch.cat(_ops{u}, dim=0)") if locals().get("cat_mode") else P(f"_st{u} = torch.stack(_ops{u}, dim={stack_dim})")
    new = [P(f"_ops{u} = []"), build, stack_stmt,
           ast.Assign(targets=[ast.Name(f"_R{u}", ast.Store())], value=stacked_call),
           ast.For(target=ast.Name(v, ast.Store()), iter=copy.deepcopy(loop.iter), orelse=[],
                   body=[P(f"{res_name} = {index_expr}")] + copy.deepcopy(post))]
    parent.body[i:i + 1] = new
    ast.fix_missing_locations(fn)
    return fn


def normalize_collectives(fn):
    """Any statement with a collective call nested inside an expression -> `_cN = xm.coll(...)` first."""
    fn = copy.deepcopy(fn)
    def fix_block(body):
        out = []
        for st in body:
            for attr in ("body", "orelse"):
                if isinstance(getattr(st, attr, None), list): setattr(st, attr, fix_block(getattr(st, attr)))
            if isinstance(st, ast.Assign) and is_coll(st.value):
                out.append(st); continue
            colls = [n for n in ast.walk(st) if is_coll(n)] if not isinstance(st, (ast.For, ast.While, ast.If)) else []
            for c in colls:
                _uid[0] += 1; tmp = f"_c{_uid[0]}"
                out.append(ast.Assign(targets=[ast.Name(tmp, ast.Store())], value=copy.deepcopy(c)))
                class Rep(ast.NodeTransformer):
                    def visit_Call(self, node):
                        if node is c: return ast.Name(tmp, ast.Load())
                        return self.generic_visit(node)
                st = Rep().visit(st)
            out.append(st)
        return out
    fn.body = fix_block(fn.body)
    ast.fix_missing_locations(fn)
    return fn

# ---------------------------------------------------------------- R2: merge two independent same-op all_reduces
def r2_sites(fn):
    out = []
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list): continue
        idx = [i for i, b in enumerate(body) if isinstance(b, ast.Assign) and is_coll(b.value) and b.value.func.attr == "all_reduce"]
        for a_i, b_i in itertools.combinations(idx, 2):
            a, b = body[a_i], body[b_i]
            if src(a.value.args[0]) != src(b.value.args[0]): continue          # same reduce op
            ta, tb = a.targets[0].id, b.targets[0].id
            between = body[a_i + 1:b_i]
            if ta in names_in(b.value): continue                                # b depends on a
            if any(tb in names_in(x) for x in between): continue                # something between uses b's result
            if any(any(is_coll(n) for n in ast.walk(x)) for x in between): continue
            # b's operand must be computable before a: its names must not be defined in `between`
            defined_between = set()
            for x in between:
                for n in ast.walk(x):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): defined_between.add(n.id)
            movers = [x for x in between if ({n.id for n in ast.walk(x) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} & names_in(b.value.args[1]))]
            if any(ta in names_in(x) for x in movers): continue                # operand prep depends on a
            out.append((parent, a_i, b_i))
    return out

def r2_apply(fn, site):
    fn = copy.deepcopy(fn); parent, a_i, b_i = site
    plist = [p for p in ast.walk(fn) if isinstance(getattr(p, "body", None), list)]
    cand = [p for p in plist if b_i < len(p.body) and src(p.body[a_i]) == src(parent.body[a_i]) and src(p.body[b_i]) == src(parent.body[b_i])]
    if not cand: return None
    parent = cand[0]; a, b = parent.body[a_i], parent.body[b_i]
    xa, xb = src(a.value.args[1]), src(b.value.args[1]); ta, tb = a.targets[0].id, b.targets[0].id
    between = parent.body[a_i + 1:b_i]
    need = set(names_in(b.value.args[1])); movers = []
    for x in reversed(between):
        defs = {n.id for n in ast.walk(x) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if defs & need: movers.insert(0, x); need |= names_in(x)
    rest = [x for x in between if x not in movers]
    _uid[0] += 1; u = _uid[0]
    merged = [P(f"_R{u} = xm.all_reduce({src(a.value.args[0])}, torch.stack([{xa}, {xb}], dim=0))"), P(f"{ta} = _R{u}[0]"), P(f"{tb} = _R{u}[1]")]
    parent.body[a_i:b_i + 1] = movers + merged + rest
    ast.fix_missing_locations(fn)
    return fn

# ---------------------------------------------------------------- R3: AG -> permute -> RS  ==>  AG -> own column
def r3_sites(fn):
    s = src(fn)
    return [0] if ("reduce_scatter" in s and "permute(1, 0, 2)" in s and "all_gather" in s) else []

def r3_apply(fn, site):
    s = src(fn)
    # find the reduce_scatter assignment and the permute line; replace both with a column slice
    m_rs = re.search(r"^(\s*)(\w+) = xm\.reduce_scatter\([^\n]*\)\n", s, re.M)
    m_tp = re.search(r"^(\s*)(\w+) = (\w+)\.permute\(1, 0, 2\)\.contiguous\(\)\.view\(-1\)\n", s, re.M)
    if not (m_rs and m_tp): return None
    indent, rs_name = m_rs.group(1), m_rs.group(2); reshaped = m_tp.group(3)
    s = s.replace(m_tp.group(0), "")
    s = s.replace(m_rs.group(0), f"{indent}{rs_name} = {reshaped}[:, rank, :].contiguous().view(-1)\n")
    try: return ast.parse(s).body[0]
    except SyntaxError: return None

# ---------------------------------------------------------------- R4: push row-reductions through all_gather(dim=1)
def r4_sites(fn):
    s = src(fn)
    return [0] if re.search(r"= xm\.all_gather\(\w+, dim=1\)", s) and (".max(" in s or ".exp()" in s or ".gather(" in s) else []

def r4_apply(fn, site):
    s = src(fn)
    m = re.search(r"^(\s*)(\w+) = xm\.all_gather\((\w+), dim=1\)\n", s, re.M)
    if not m: return None
    indent, g, local = m.groups()
    s2 = s.replace(m.group(0), "")
    s2 = re.sub(rf"(\w+) = {g}\.max\(dim=-1(, keepdim=True)?\)\.values", rf"\1 = xm.all_reduce(xm.REDUCE_MAX, {local}.max(dim=-1\2).values)", s2)
    s2 = re.sub(rf"(\w+) = {g} - (\w+)\b", rf"\1 = {local} - \2", s2)
    s2 = re.sub(rf"(\w+) = (\w+)\.exp\(\)\.sum\(dim=-1\)\.log\(\)", rf"\1 = xm.all_reduce(xm.REDUCE_SUM, \2.exp().sum(dim=-1)).log()", s2)
    s2 = re.sub(rf"(\w+) = {g}\.gather\(1, (\w+)\.unsqueeze\(1\)\)\.squeeze\(1\)",
                rf"_lo = rank * V_local\n{indent}_in = (\2 >= _lo) & (\2 < _lo + V_local)\n{indent}_t = {local}.gather(1, (\2 - _lo).clamp(0, V_local - 1).unsqueeze(1)).squeeze(1)\n{indent}\1 = xm.all_reduce(xm.REDUCE_SUM, torch.where(_in, _t, torch.zeros_like(_t)))", s2)
    if re.search(rf"\b{g}\b", s2): return None
    try: return ast.parse(s2).body[0]
    except SyntaxError: return None

RULES = {"R1_hoist_loop": (r1_sites, r1_apply), "R2_merge_adjacent": (r2_sites, r2_apply),
         "R3_ag_rs_to_slice": (r3_sites, r3_apply), "R4_reduce_through_gather": (r4_sites, r4_apply)}

# ---------------------------------------------------------------- oracle + objective
def compile_fn(problem, code):
    ns = {}; exec(code, ns); fn = ns[problem.evolved_fn_name]; fn.__candidate_source__ = code; return fn

def correct(problem, fn):
    try:
        r = test_xla_candidate_generic(problem, fn, world_sizes=[8], verbose=False)
        if isinstance(r, tuple):
            if DEBUG and not r[0]: print(f"      oracle: {str(r[1])[:160]}", flush=True)
            return bool(r[0])
        if isinstance(r, dict): return bool(r.get("passed", r.get("all_passed", False)))
        return bool(r)
    except Exception:
        return False


# ---------------------------------------------------------------- real-tensor multi-rank oracle (threads + barriers)
import threading
class _Barrier:
    def __init__(self, n): self.n = n; self.b = threading.Barrier(n); self.lock = threading.Lock(); self.slots = {}; self.gen = 0
class ThreadedXM:
    REDUCE_SUM = "sum"; REDUCE_MAX = "max"; REDUCE_MIN = "min"
    def __init__(self, shared, rank, ws): self.sh, self.rank, self.ws = shared, rank, ws
    def _exchange(self, x):
        sh = self.sh
        with sh.lock: sh.slots[self.rank] = x
        sh.b.wait()
        vals = [sh.slots[r] for r in range(self.ws)]
        sh.b.wait()
        return vals
    def all_gather(self, x, dim=0, **kw): return torch.cat(self._exchange(x), dim=dim)
    def all_reduce(self, op, x, scale=1.0, **kw):
        vals = self._exchange(x)
        if op == "sum": r = sum(vals[1:], vals[0].clone())
        elif op == "max": r = torch.stack(vals).max(dim=0).values
        else: r = torch.stack(vals).min(dim=0).values
        return r * scale if scale != 1.0 else r
    def reduce_scatter(self, op, x, scale=1.0, scatter_dim=0, shard_count=None, **kw):
        full = self.all_reduce(op, x, scale=scale)
        return full.chunk(self.ws, dim=scatter_dim)[self.rank].contiguous()
    def mark_step(self): pass
    def rendezvous(self, *a): pass

def real_oracle(problem, fn, ws=8, pattern="uniform"):
    """Run all ranks as threads with true collectives; compare to the problem's expected outputs."""
    import torch as _torch
    try:
        tc = problem.generate_test_case(ws, pattern, 32, seed=0)
    except TypeError:
        tc = problem.generate_test_case(ws, pattern)
    shared = _Barrier(ws); outs = [None] * ws; errs = []
    def worker(r):
        try:
            xm = ThreadedXM(shared, r, ws)
            o = problem.call_candidate(fn, tc["per_rank_args"][r], tc["shared_args"], r, ws, ws // 2, 2, xm, _torch, num_nodes=1)
            outs[r] = o
        except Exception as e:
            errs.append(f"rank{r}: {type(e).__name__}: {str(e)[:120]}"); shared.b.abort()
    ts = [threading.Thread(target=worker, args=(r,)) for r in range(ws)]
    for t in ts: t.start()
    for t in ts: t.join(timeout=120)
    if errs: return False, errs[0]
    from search.correctness_test import _compare_outputs_generic
    return _compare_outputs_generic(outs, tc["expected"], ws, pattern)

ORACLE_LOG = []
_evals = {"n": 0}
def sim(problem, fn):
    _evals["n"] += 1
    b = benchmark_xla_candidate_generic(problem, fn, TOPO, SEND, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"])
    return None if "error" in b else b["sim_time_us"]

def expand(fn_ast):
    fn_ast = normalize_collectives(fn_ast)
    for rname, (sites, apply) in RULES.items():
        for site in sites(fn_ast):
            try: new = apply(fn_ast, site)
            except Exception: new = None
            if new is not None: yield rname, new

def beam(problem, start_ast, width=3, depth=4, budget=BUDGET):
    _evals["n"] = 0
    f0 = compile_fn(problem, src(start_ast)); s0 = sim(problem, f0)
    frontier = [(s0, start_ast, [])]; best = (s0, src(start_ast), []); seen = {src(start_ast)}
    for d in range(depth):
        cands = []
        for s, a, hist in frontier:
            for rname, new in expand(a):
                code = src(new)
                if code in seen: continue
                seen.add(code)
                try: fn = compile_fn(problem, code)
                except Exception as e:
                    if DEBUG: print(f"    [{rname}] compile fail: {str(e)[:80]}", flush=True)
                    continue
                ok = correct(problem, fn); via = "sandbox" if ok else None
                if not ok:
                    ok2, msg = real_oracle(problem, fn)
                    if ok2: ok, via = True, "real-tensor"
                    elif DEBUG: print(f"      real-oracle: {msg}", flush=True)
                v = sim(problem, fn) if ok else None
                ORACLE_LOG.append(dict(rule=rname, accepted=ok, via=via, sim=v))
                if DEBUG: print(f"    [{rname}] correct={ok} via={via} sim={v}", flush=True)
                if DEBUG and not ok and os.environ.get("DUMP"): print(code, flush=True)
                if not ok or v is None: continue
                cands.append((v, new, hist + [rname]))
                if _evals["n"] >= budget: break
            if _evals["n"] >= budget: break
        if not cands: break
        cands.sort(key=lambda t: t[0]); frontier = cands[:width]
        if cands[0][0] < best[0]: best = (cands[0][0], src(cands[0][1]), cands[0][2])
        if _evals["n"] >= budget: break
    return dict(best_sim=best[0], start_sim=s0, rules=best[2], code=best[1], evals=_evals["n"])

def random_walk(problem, start_ast, budget=BUDGET, steps=4):
    _evals["n"] = 0; f0 = compile_fn(problem, src(start_ast)); s0 = sim(problem, f0); best = (s0, src(start_ast), [])
    while _evals["n"] < budget:
        a, hist = start_ast, []
        for _ in range(steps):
            moves = list(expand(a))
            if not moves: break
            rname, a = random.choice(moves); hist.append(rname)
        code = src(a)
        try: fn = compile_fn(problem, code)
        except Exception: continue
        if not correct(problem, fn): _evals["n"] += 1; continue
        v = sim(problem, fn)
        if v is not None and v < best[0]: best = (v, code, hist)
    return dict(best_sim=best[0], start_sim=s0, rules=best[2], code=best[1], evals=_evals["n"])

def structure(code):
    """coarse signature: collective op counts (static) for comparing against the deployed strategy"""
    t = ast.parse(code); c = {}
    for n in ast.walk(t):
        if is_coll(n): c[n.func.attr] = c.get(n.func.attr, 0) + 1
    loops = sum(isinstance(n, ast.For) for n in ast.walk(t))
    return dict(colls=c, loops=loops)

if __name__ == "__main__":
    out = {}
    for pname, bname in BASELINE.items():
        if ONLY and pname not in ONLY: continue
        problem = get_problem(pname); code = RING_KV_BASELINE if pname == 'ring_kv' else problem.builtin_templates[bname]
        start = ast.parse(code).body[0]
        t0 = time.time(); ORACLE_LOG.clear()
        try: b = beam(problem, start)
        except Exception as e: b = dict(error=str(e)[:200])
        try: r = random_walk(problem, start)
        except Exception as e: r = dict(error=str(e)[:200])
        out[pname] = dict(baseline=bname, beam=b, random_walk=r, oracle_log=list(ORACLE_LOG), start_structure=structure(code),
                          beam_structure=structure(b["code"]) if "code" in b else None, secs=round(time.time() - t0, 1))
        print(f"{pname:15s} start={b.get('start_sim')} beam={b.get('best_sim')} rules={b.get('rules')} evals={b.get('evals')} | rw={r.get('best_sim')} {r.get('rules')} | struct {out[pname]['start_structure']} -> {out[pname]['beam_structure']}", flush=True)
    os.makedirs("/home/ubuntu/sweep_results", exist_ok=True)
    json.dump(out, open("/home/ubuntu/sweep_results/rewriter_search.json", "w"), indent=1)
    print("wrote rewriter_search.json")
