"""Batch-validate v6 (rewritten broad/diverse set): for each problem confirm
baseline passes the fp32 gate, then confirm the intended optimum ALSO passes
and measure sim headroom (baseline/opt). Keep only problems with real headroom.
"""
import contextlib, os, sys
ACC = "/private/tmp/acc_verify"; sys.path.insert(0, ACC)
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


def _ar_whole(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            "    return xm.all_reduce(xm.REDUCE_SUM, x)\n")


def _ar_live(name, part=256):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            f"    return xm.all_reduce(xm.REDUCE_SUM, x[:{part}])\n")


def _cse(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            "    return 2.0 * xm.all_reduce(xm.REDUCE_SUM, x)\n")


def _alg(name, coeffs, part=256):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            f"    part = {part}; coeffs = {list(coeffs)}\n"
            "    acc = None\n"
            "    for j, c in enumerate(coeffs):\n"
            "        seg = x[j*part:(j+1)*part]\n"
            "        acc = c*seg if acc is None else acc + c*seg\n"
            "    return xm.all_reduce(xm.REDUCE_SUM, acc)\n")


def _scale(name, coeffs):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            f"    return {float(sum(coeffs))} * xm.all_reduce(xm.REDUCE_SUM, x)\n")


def _a2a(name):
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            "                 cores_per_device, xm, torch, num_nodes=1):\n"
            "    return xm.all_to_all(x, split_dimension=0, concat_dimension=0,\n"
            "                         split_count=world_size)\n")


OPT = {
    "v6_fuse3": _ar_whole("v6_fuse3"),
    "v6_fuse6": _ar_whole("v6_fuse6"),
    "v6_fuse10": _ar_whole("v6_fuse10"),
    "v6_dead1": _ar_live("v6_dead1"),
    "v6_dead3": _ar_live("v6_dead3"),
    "v6_dead5": _ar_live("v6_dead5"),
    "v6_cse2": _cse("v6_cse2"),
    "v6_alg2": _alg("v6_alg2", (2.0, -1.0)),
    "v6_alg3": _alg("v6_alg3", (1.5, -0.5, 2.0)),
    "v6_alg5": _alg("v6_alg5", (1.0, -1.0, 2.0, 0.5, -1.5)),
    "v6_a2a_k128": _a2a("v6_a2a_k128"),
    "v6_a2a_k256": _a2a("v6_a2a_k256"),
    "v6_scale3": _scale("v6_scale3", (1.0, 2.0, 3.0)),
    "v6_scale6": _scale("v6_scale6", (1.0, -1.0, 2.0, 0.5, 1.5, -0.5)),
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


def bench_kw(problem):
    _ts = RS._problem_train_scale_multiplier(problem)
    return dict(
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


print(f"{'problem':16}{'base_ok':>8}{'base_sim':>11}{'opt_ok':>8}{'opt_sim':>11}{'headroom':>9}")
for name in OPT:
    problem = get_problem(name)
    unsupported = list(getattr(agent_sim.config, "unsupported_primitives", []) or [])
    BK = bench_kw(problem)
    te = TemplateEvolution(topology, send_counts, agent_sim,
                           ContentionAnalyzer(topology, send_counts),
                           model="sonnet", problem=problem,
                           unsupported_primitives=unsupported)
    base_code = problem.builtin_templates[name]
    with contextlib.redirect_stdout(sys.stderr):
        try:
            bfn = te._sandbox_exec(base_code, is_nki=False)
            bok, bdet = test_xla_candidate_generic(problem, bfn, num_nodes=NODES,
                                                   unsupported_primitives=unsupported)
            bbench = benchmark_xla_candidate_generic(
                problem, bfn, topology, send_counts, world, num_nodes=NODES, **BK) if bok else None
        except Exception as e:
            bok, bdet, bbench = False, repr(e)[:80], None
        try:
            ofn = te._sandbox_exec(OPT[name], is_nki=False)
            ook, odet = test_xla_candidate_generic(problem, ofn, num_nodes=NODES,
                                                   unsupported_primitives=unsupported)
            obench = benchmark_xla_candidate_generic(
                problem, ofn, topology, send_counts, world, num_nodes=NODES, **BK) if ook else None
        except Exception as e:
            ook, odet, obench = False, repr(e)[:80], None
    bs = bbench.get("sim_time_us") if bbench else None
    os_ = obench.get("sim_time_us") if obench else None
    hr = round(bs/os_, 3) if (bs and os_) else None
    print(f"{name:16}{str(bok):>8}{(f'{bs:.1f}' if bs else '-'):>11}"
          f"{str(ook):>8}{(f'{os_:.1f}' if os_ else '-'):>11}{str(hr):>9}")
    if not bok: print(f"    base fail: {str(bdet)[:120]}")
    if not ook: print(f"    opt  fail: {str(odet)[:120]}")
