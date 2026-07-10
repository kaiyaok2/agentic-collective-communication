"""Long-running scorer service (Neuron venv, Python 3.12). Replicates
cc-react's exact Phase 1+2 setup (no-LLM Phase 1) so kiss candidates are
scored under the same cost function.

Environment variables:
  ACC_REPO         Root of the agentic-collective-communication repo.
                   Defaults to two directories up from this file.
  SCORE_PROBLEM    Required. Problem name (e.g. 'alltoallv', 'dxe').
  SCORE_PATTERN    Required. Send-count pattern (e.g. 'moe', 'uniform').
  SCORE_NUM_NODES  Optional (default 1).

Wire protocol:
  stdin  : one JSON object per line, {"code": "..."} or {"cmd": "quit"}.
  stdout : one JSON response per line (only score results).
  stderr : init logs.
"""
import contextlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.environ.get(
    "ACC_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, ACC)
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

PROBLEM = os.environ["SCORE_PROBLEM"]
PATTERN = os.environ["SCORE_PATTERN"]
NUM_NODES = int(os.environ.get("SCORE_NUM_NODES", "1"))

# Redirect all init prints to stderr so stdout is reserved for JSON responses.
with contextlib.redirect_stdout(sys.stderr):
    import experiments.run_search as RS
    from search.problems import get_problem
    from search.template_evolution import TemplateEvolution
    from search.contention_analysis import ContentionAnalyzer
    from search.correctness_test import (
        test_xla_candidate_generic, benchmark_xla_candidate_generic,
    )

    agent_sim, topology, dispatch_overhead = RS.phase1_profiling(
        use_llm=False, llm_model="sonnet", num_nodes=NUM_NODES, verbose=False)
    world = topology.num_cores
    send_counts = RS.make_send_counts(PATTERN, world=world)
    problem = get_problem(PROBLEM)

    # Build the same bench kwargs cc-react uses (see run_search.main lines ~2270).
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

# Init done. Signal ready.
print(f"[score_service] ready for {PROBLEM}/{PATTERN}/nn={NUM_NODES}",
      file=sys.stderr, flush=True)

# Request loop.
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
                else:
                    bench = benchmark_xla_candidate_generic(
                        problem, fn, topology, send_counts, world,
                        num_nodes=NUM_NODES, **BENCH_KW)
                    out = {"ok": True,
                           "sim_time_us": float(bench.get("sim_time_us", 0)),
                           "num_ops": int(bench.get("local_ops", 0) or 0),
                           "num_all_gather": int(bench.get("num_all_gather", 0)),
                           "num_all_reduce": int(bench.get("num_all_reduce", 0)),
                           "num_collective_permute": int(bench.get("num_collective_permute", 0))}
        except Exception:
            out = {"ok": False, "error": traceback.format_exc()[-500:]}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
