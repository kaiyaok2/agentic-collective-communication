"""Long-running scorer service v2 — with optional HW correctness gate.

Env vars (all backward-compat):
  ACC_REPO         Root of the agentic-collective-communication repo.
  SCORE_PROBLEM    Required. Problem name.
  SCORE_PATTERN    Required. Send-count pattern.
  SCORE_NUM_NODES  Optional (default 1).
  HW_GATE          "1" to enable HW correctness gate (default off).
  HW_GATE_SCRIPT   path to hw_gate_run.py (default /home/ubuntu/hw_gate_run.py).
  HW_GATE_MODE     "improved" (default) or "all". Only gate candidates whose
                   sim_time_us < best_seen when "improved".
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
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

HW_GATE = os.environ.get("HW_GATE") == "1"
HW_GATE_SCRIPT = os.environ.get("HW_GATE_SCRIPT", "/home/ubuntu/hw_gate_run.py")
HW_GATE_MODE = os.environ.get("HW_GATE_MODE", "improved")
_best_sim_seen = [float("inf")]

with contextlib.redirect_stdout(sys.stderr):
    import experiments.run_search as RS
    from search.problems import get_problem
    import search.problems_kiss_verify
    import search.problems_novel_v4
    import search.problems_novel_v5
    import search.problems_novel_v6
    import search.problems_comm_v7
import search.problems_challenge_v8
    import search.problems_modext
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
        standalone_graph_cost_cfg=agent_sim.knowledgebase.get("standalone_graph_cost_us", {}),
        unsupported_local_ops=[p for p in (agent_sim.config.unsupported_primitives or []) if p in {"cumsum", "cumprod", "sort", "argsort"}],
    )
    te = TemplateEvolution(topology, send_counts, agent_sim,
                            ContentionAnalyzer(topology, send_counts),
                            model="sonnet", problem=problem,
                            unsupported_primitives=unsupported)


def run_hw_gate(code: str) -> tuple:
    """Run 32-rank single-node hw_gate check. Returns (ok, err_str)."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w")
    tf.write(code); tf.close()
    try:
        env = os.environ.copy()
        env.update({
            "PATH": "/home/ubuntu/venv/bin:" + env.get("PATH", ""),
            "LD_LIBRARY_PATH": "/opt/aws/neuron/lib",
            "PJRT_DEVICE": "NEURON",
            "FI_PROVIDER": "efa",
            "FI_EFA_USE_DEVICE_RDMA": "1",
        })
        MASTER_IP = os.environ.get("MASTER_IP", "172.31.27.131")
        WORKER_IP = os.environ.get("WORKER_IP", "172.31.28.247")
        result = subprocess.run(
            ["/bin/bash", "/home/ubuntu/hw_gate_2node.sh",
             tf.name, PROBLEM, MASTER_IP, WORKER_IP],
            capture_output=True, timeout=300, env=env)
        if result.returncode == 0:
            return True, ""
        stderr_full = result.stderr.decode()
        # Extract structured diagnostics from hw_gate_run.py output
        diag_lines = []
        for line in stderr_full.splitlines():
            if any(k in line for k in ["SHAPE_MISMATCH", "VALUE_MISMATCH",
                                        "HW_GATE_EXCEPTION", "NO_EVOLVED_FN",
                                        "Logical Neuron Core", "Bad StatusOr"]):
                diag_lines.append(line.strip())
        # Also grab any short exception message (last line before empty)
        if not diag_lines:
            # Search for a torch error signature
            for line in stderr_full.splitlines():
                if "RuntimeError:" in line or "Error:" in line or "failed with" in line:
                    diag_lines.append(line.strip()[:200])
                    break
        if diag_lines:
            # De-dup and keep first ~3 unique diagnostic lines (all ranks may print similar)
            seen = set(); uniq = []
            for l in diag_lines:
                key = l.split("rank=")[0]  # strip rank number for dedup
                if key not in seen and len(uniq) < 4:
                    seen.add(key); uniq.append(l)
            return False, "HW_GATE_FAIL: " + " | ".join(uniq)
        return False, (stderr_full[-400:] or "unknown_fail")
    except subprocess.TimeoutExpired:
        return False, "HW_GATE_TIMEOUT"
    except Exception as e:
        return False, f"HW_GATE_LAUNCH_FAIL: {e}"
    finally:
        try: os.unlink(tf.name)
        except: pass


print(f"[score_service_v2] ready for {PROBLEM}/{PATTERN}/nn={NUM_NODES} "
      f"(HW_GATE={'on' if HW_GATE else 'off'})",
      file=sys.stderr, flush=True)

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
                    sim_us = float(bench.get("sim_time_us", 0))
                    out = {"ok": True,
                           "sim_time_us": sim_us,
                           "num_ops": int(bench.get("local_ops", 0) or 0),
                           "num_all_gather": int(bench.get("num_all_gather", 0)),
                           "num_all_reduce": int(bench.get("num_all_reduce", 0)),
                           "num_collective_permute": int(bench.get("num_collective_permute", 0))}
                    # HW gate: only if enabled AND candidate improves leader
                    if HW_GATE and (HW_GATE_MODE == "all" or sim_us < _best_sim_seen[0]):
                        print(f"[hw_gate] running for sim_time_us={sim_us:.2f} "
                              f"(best_seen={_best_sim_seen[0]:.2f})",
                              file=sys.stderr, flush=True)
                        gate_ok, gate_err = run_hw_gate(code)
                        if not gate_ok:
                            out = {"ok": False,
                                   "error": f"HW_GATE_FAIL: {gate_err}"}
                        else:
                            if sim_us < _best_sim_seen[0]:
                                _best_sim_seen[0] = sim_us
                            print(f"[hw_gate] pass; new best={sim_us:.2f}",
                                  file=sys.stderr, flush=True)
        except Exception:
            out = {"ok": False, "error": traceback.format_exc()[-500:]}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
