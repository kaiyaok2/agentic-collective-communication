"""Fair symmetric scorer service. IDENTICAL correctness gate for BOTH
pipelines (OverlayCCL strategy-enumerate and SorcarCCL kiss-ReAct).

The whole point of this harness: the bf16-gate asymmetry in the shipped
repo (strat runs test_xla_candidate_bf16, kiss's score_service.py does
not) manufactured the 55-problem divergence. Here BOTH sides call this
one service, so the correctness gate is provably identical. GATE_MODE
selects which symmetric gate to apply:

  GATE_MODE=fp32       -> test_xla_candidate_generic only (atol=1e-5)
  GATE_MODE=fp32_bf16  -> also require test_xla_candidate_bf16 (atol=0.1)

Any divergence observed under a FIXED GATE_MODE is a search-shape effect,
not a gate artifact.

Env (same as score_service.py) + GATE_MODE.
Wire protocol identical to score_service.py: one JSON {"code": ...} per
line in, one JSON response per line out.
"""
import contextlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.environ.get("ACC_REPO", "/private/tmp/acc_verify")
sys.path.insert(0, ACC)
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

PROBLEM = os.environ["SCORE_PROBLEM"]
PATTERN = os.environ["SCORE_PATTERN"]
NUM_NODES = int(os.environ.get("SCORE_NUM_NODES", "7"))
GATE_MODE = os.environ.get("GATE_MODE", "fp32")  # fp32 | fp32_bf16

with contextlib.redirect_stdout(sys.stderr):
    import experiments.run_search as RS
    from search.problems import get_problem
    import search.problems_all_catalogs  # noqa: register all catalogs
    from search.template_evolution import TemplateEvolution
    from search.contention_analysis import ContentionAnalyzer
    from search.correctness_test import (
        test_xla_candidate_generic, test_xla_candidate_bf16,
        benchmark_xla_candidate_generic,
    )

    agent_sim, topology, dispatch_overhead = RS.phase1_profiling(
        use_llm=False, llm_model="sonnet", num_nodes=NUM_NODES, verbose=False)
    world = topology.num_cores
    send_counts = RS.make_send_counts(PATTERN, world=world)
    problem = get_problem(PROBLEM)

    op_costs = RS._extract_op_costs(agent_sim)
    _HM = RS._HARDWARE_MEASUREMENTS or {}
    _comp_info = _HM.get("compilation_cost_us", {}) or {}
    _comp_samples = _comp_info.get("samples", [])
    _load_events = _comp_info.get("load_events_per_run", 2)
    _amortize = _comp_info.get("typical_training_steps", 5000)
    _glo = _HM.get("graph_launch_overhead_us", {}) or {}
    _glo_us = _glo.get("per_mark_step_us", 0.0)
    _train_scale = RS._problem_train_scale_multiplier(problem)
    _memcpy_bps, _memcpy_seq_bps = RS._extract_memcpy_bw(agent_sim)
    _mc = _HM.get("memcpy_throughput", {}) or {}
    if _memcpy_bps == 0.0:
        _memcpy_bps = float(_mc.get("strided_gbps", 0.0)) * 1000.0
    if _memcpy_seq_bps == 0.0:
        _memcpy_seq_bps = float(_mc.get("sequential_gbps", 0.0)) * 1000.0
    _a1 = getattr(agent_sim.config, "pipeline_amort_alpha1", 0.30)
    _a2 = getattr(agent_sim.config, "pipeline_amort_alpha2", 0.10)
    _a3 = getattr(agent_sim.config, "pipeline_amort_alpha3", 0.02)
    unsupported = list(getattr(agent_sim.config, "unsupported_primitives", []) or [])

    BENCH_KW = dict(
        op_costs=op_costs,
        dispatch_overhead_us=dispatch_overhead,
        graph_launch_overhead_us=_glo_us,
        compilation_cost_samples=_comp_samples,
        compilation_load_events_per_run=_load_events,
        compilation_amortize_steps=_amortize,
        pipeline_amort_alpha1=_a1,
        pipeline_amort_alpha2=_a2,
        pipeline_amort_alpha3=_a3,
        training_scale_bytes_multiplier=_train_scale,
        memcpy_bytes_per_us=_memcpy_bps,
        memcpy_seq_bytes_per_us=_memcpy_seq_bps,
    )
    te = TemplateEvolution(topology, send_counts, agent_sim,
                           ContentionAnalyzer(topology, send_counts),
                           model="sonnet", problem=problem,
                           unsupported_primitives=unsupported)

print(f"[score_service] ready for {PROBLEM}/{PATTERN}/nn={NUM_NODES} "
      f"gate={GATE_MODE}", file=sys.stderr, flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        sys.stdout.write(json.dumps({"ok": False, "error": "bad json"}) + "\n")
        sys.stdout.flush()
        continue
    if req.get("cmd") == "quit":
        break
    code = req.get("code", "")
    with contextlib.redirect_stdout(sys.stderr):
        try:
            fn = te._sandbox_exec(code, is_nki=False)
            if fn is None:
                out = {"ok": False, "error": "sandbox_exec returned None"}
            else:
                passed, details = test_xla_candidate_generic(
                    problem, fn, num_nodes=NUM_NODES,
                    unsupported_primitives=unsupported)
                if not passed:
                    out = {"ok": False,
                           "error": f"correctness fail: {str(details)[:200]}"}
                elif GATE_MODE == "fp32_bf16":
                    bf16_ok, bf16_details = test_xla_candidate_bf16(
                        problem, fn, num_nodes=NUM_NODES,
                        unsupported_primitives=unsupported)
                    if not bf16_ok:
                        out = {"ok": False,
                               "error": f"bf16 fail: {str(bf16_details)[:200]}"}
                    else:
                        out = None
                else:
                    out = None
                if out is None:
                    bench = benchmark_xla_candidate_generic(
                        problem, fn, topology, send_counts, world,
                        num_nodes=NUM_NODES, **BENCH_KW)
                    sim_us = float(bench.get("sim_time_us", 0))
                    n_ag = int(bench.get("num_all_gather", 0))
                    n_ar = int(bench.get("num_all_reduce", 0))
                    n_cp = int(bench.get("num_collective_permute", 0))
                    n_rs = int(bench.get("num_reduce_scatter", 0) or 0)
                    n_coll = n_ag + n_ar + n_cp + n_rs
                    # SYMMETRIC scorer-artifact guard (applies to BOTH
                    # pipelines identically): every problem in this study
                    # requires genuine cross-rank communication, so a
                    # CORRECT candidate cannot cost below the single-
                    # collective floor (~5160us). The known
                    # reduce_scatter+all_gather pattern is mis-traced by the
                    # cost model to sim_time_us ~= 0 with zeroed op/collective
                    # counts -- a scorer bug, not a real optimum. Left
                    # unguarded it manufactures a FALSE divergence whenever
                    # one pipeline stumbles into it. Reject anything under a
                    # conservative 1000us physical floor rather than let it
                    # win. (No legitimate candidate here lands there.)
                    if sim_us < 1000.0:
                        out = {"ok": False,
                               "error": (f"scorer artifact: sim={sim_us:.1f}us "
                                         f"below physical floor with n_coll="
                                         f"{n_coll}; rejected symmetrically")}
                    else:
                        out = {"ok": True, "sim_time_us": sim_us,
                               "num_ops": int(bench.get("local_ops", 0) or 0),
                               "num_all_gather": n_ag, "num_all_reduce": n_ar,
                               "num_collective_permute": n_cp,
                               "num_reduce_scatter": n_rs}
        except Exception:
            out = {"ok": False, "error": traceback.format_exc()[-500:]}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
