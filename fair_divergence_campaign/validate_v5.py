"""Validate v5 problems: (1) builtin baseline passes the fp32 gate,
(2) the intended 1-collective optimum ALSO passes the same gate,
(3) measure sim headroom baseline vs optimum. Uses the SAME scorer machinery.
"""
import contextlib
import os
import sys

ACC = "/private/tmp/acc_verify"
sys.path.insert(0, ACC)
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
NODES = 7

with contextlib.redirect_stdout(sys.stderr):
    import experiments.run_search as RS
    from search.problems import get_problem
    import search.problems_all_catalogs  # noqa
    from search.template_evolution import TemplateEvolution
    from search.contention_analysis import ContentionAnalyzer
    from search.correctness_test import (
        test_xla_candidate_generic, benchmark_xla_candidate_generic)

    agent_sim, topology, dispatch = RS.phase1_profiling(
        use_llm=False, llm_model="sonnet", num_nodes=NODES, verbose=False)
    world = topology.num_cores

# The optimum code, per problem. coeff * ONE all_reduce(SUM, x).
S = 256
OPT = {
"hd16_product_scale_chain": f"""
def hd16_product_scale_chain_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = {S}; W = world_size
    A = [[1.0 + 0.3*(r%4) for r in range(W)],
         [0.5 + 0.2*((r+1)%3) for r in range(W)],
         [0.75 + 0.1*(r%5) for r in range(W)],
         [1.25 - 0.05*(r%6) for r in range(W)]]
    coeff = torch.zeros(W*S, device=x.device, dtype=x.dtype)
    for r in range(W):
        p = 1.0
        for j in range(4):
            p *= A[j][r]
        coeff[r*S:(r+1)*S] = p
    return coeff * xm.all_reduce(xm.REDUCE_SUM, x)
""",
"hd17_intrablock_ramp_chain": f"""
def hd17_intrablock_ramp_chain_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = {S}; W = world_size; beta = 0.01
    ramp = 1.0 + beta * torch.arange(S, dtype=x.dtype)
    coeff = torch.zeros(W*S, device=x.device, dtype=x.dtype)
    for r in range(W):
        c = 0.5 + 0.25*(r%4)
        coeff[r*S:(r+1)*S] = (c*ramp)**2
    return coeff * xm.all_reduce(xm.REDUCE_SUM, x)
""",
"hd18_sign_telescope_chain": f"""
def hd18_sign_telescope_chain_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = {S}; W = world_size
    coeff = torch.zeros(W*S, device=x.device, dtype=x.dtype)
    for r in range(W):
        mag = (1.0 + 0.2*(r%3)) * (0.8 + 0.1*(r%4)) * (1.1 - 0.05*(r%5))
        sgn = 1.0 if (r%2==0) else -1.0
        coeff[r*S:(r+1)*S] = sgn*mag
    return coeff * xm.all_reduce(xm.REDUCE_SUM, x)
""",
"hd19_mixed_sum_max_chain": f"""
def hd19_mixed_sum_max_chain_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = {S}; W = world_size
    coeff = torch.zeros(W*S, device=x.device, dtype=x.dtype)
    for r in range(W):
        coeff[r*S:(r+1)*S] = 1.0 + 0.5*(r%3)
    return coeff * xm.all_reduce(xm.REDUCE_SUM, x)
""",
"hd20_modular_coeff_chain": f"""
def hd20_modular_coeff_chain_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = {S}; W = world_size; m = 7
    base = [0.5, 1.0, 1.5, 0.75, 1.25, 0.9, 1.1]
    N = W*S
    idx = torch.arange(N) % m
    coeff = torch.tensor([base[int(k)] for k in idx], dtype=x.dtype)
    return (coeff**2) * xm.all_reduce(xm.REDUCE_SUM, x)
""",
}

send_counts = RS.make_send_counts("moe", world=world)
op_costs = RS._extract_op_costs(agent_sim)
_HM = RS._HARDWARE_MEASUREMENTS or {}
_comp = _HM.get("compilation_cost_us", {}) or {}
_glo = (_HM.get("graph_launch_overhead_us", {}) or {}).get("per_mark_step_us", 0.0)
_mcb, _mcs = RS._extract_memcpy_bw(agent_sim)
_mc = _HM.get("memcpy_throughput", {}) or {}
if _mcb == 0.0: _mcb = float(_mc.get("strided_gbps", 0.0)) * 1000.0
if _mcs == 0.0: _mcs = float(_mc.get("sequential_gbps", 0.0)) * 1000.0

for name in OPT:
    problem = get_problem(name)
    _ts = RS._problem_train_scale_multiplier(problem)
    unsupported = list(getattr(agent_sim.config, "unsupported_primitives", []) or [])
    BENCH_KW = dict(
        op_costs=op_costs, dispatch_overhead_us=dispatch,
        graph_launch_overhead_us=_glo,
        compilation_cost_samples=_comp.get("samples", []),
        compilation_load_events_per_run=_comp.get("load_events_per_run", 2),
        compilation_amortize_steps=_comp.get("typical_training_steps", 5000),
        pipeline_amort_alpha1=getattr(agent_sim.config, "pipeline_amort_alpha1", 0.30),
        pipeline_amort_alpha2=getattr(agent_sim.config, "pipeline_amort_alpha2", 0.10),
        pipeline_amort_alpha3=getattr(agent_sim.config, "pipeline_amort_alpha3", 0.02),
        training_scale_bytes_multiplier=_ts,
        memcpy_bytes_per_us=_mcb, memcpy_seq_bytes_per_us=_mcs)
    te = TemplateEvolution(topology, send_counts, agent_sim,
                           ContentionAnalyzer(topology, send_counts),
                           model="sonnet", problem=problem,
                           unsupported_primitives=unsupported)

    # baseline (builtin)
    base_code = problem.builtin_templates[name]
    with contextlib.redirect_stdout(sys.stderr):
        bfn = te._sandbox_exec(base_code, is_nki=False)
        bok, bdet = test_xla_candidate_generic(problem, bfn, num_nodes=NODES,
                                               unsupported_primitives=unsupported)
        bbench = benchmark_xla_candidate_generic(
            problem, bfn, topology, send_counts, world, num_nodes=NODES, **BENCH_KW) if bok else None
        # optimum
        ofn = te._sandbox_exec(OPT[name], is_nki=False)
        ook, odet = test_xla_candidate_generic(problem, ofn, num_nodes=NODES,
                                               unsupported_primitives=unsupported)
        obench = benchmark_xla_candidate_generic(
            problem, ofn, topology, send_counts, world, num_nodes=NODES, **BENCH_KW) if ook else None
    bs = bbench.get("sim_time_us") if bbench else None
    os_ = obench.get("sim_time_us") if obench else None
    ratio = round(bs/os_, 3) if (bs and os_) else None
    print(f"{name}")
    print(f"  baseline: ok={bok} sim={bs}  ({str(bdet)[:80] if not bok else ''})")
    print(f"  optimum : ok={ook} sim={os_}  ({str(odet)[:80] if not ook else ''})")
    print(f"  headroom (baseline/opt) = {ratio}")
