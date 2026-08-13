#!/usr/bin/env python3
"""
5-Phase AllToAllV Search Pipeline for AWS Trainium.

Hardware-agnostic agent workflow that discovers performance characteristics
through profiling and evolves optimal AllToAllV implementations.

Phase 1: Agent Hardware Profiling — LLM builds its own simulator
Phase 2: Baseline Evaluation on Simulator → Knowledgebase
Phase 3: Multi-island Evolution with Simulator Feedback
Phase 4: Iterative Mini-benchmarking + Refining on Real HW
Phase 5: Final Code Generation → runtime/trainium_alltoallv.py

Feedback loops: Phase 2 and 3 can return to Phase 1 to refine the simulator
when predictions diverge from observed behavior.

Usage:
    python experiments/run_search.py --pattern moe --no-llm
    python experiments/run_search.py --pattern skewed --llm-model haiku --hw-eval
    python experiments/run_search.py --all-patterns --emit-cpp
"""

import argparse
import multiprocessing as _mp
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.topology import TrainiumTopology, MultiNodeTopology
from simulator.alltoallv import AllToAllVSimulator
from simulator.cost_model import CostModel
from search.templates import (
    TEMPLATES, TemplateConfig,
    permute_ring_default_params,
    allgather_slice_default_params,
    hierarchical_default_params,
    pairwise_default_params,
    hybrid_default_params,
    _generate_matchings,
)
from search.generate_algo import (
    generate_llm_schedule,
    topology_aware_schedule,
    butterfly_schedule,
    contention_aware_schedule,
    traffic_adaptive_schedule,
    genetic_search,
    simulated_annealing,
    local_search,
)
from search.evaluate_algo import (
    evaluate_template,
    generate_trainium_code,
    run_on_hardware,
)
from search.contention_analysis import ContentionAnalyzer
from search.iterative_refinement import IterativeRefinement
from search.island_evolution import IslandEvolution
from search.template_evolution import TemplateEvolution
from search.problems import get_problem, PROBLEMS
import search.problems_kiss_verify  # noqa
import search.problems_modext  # noqa
import search.problems_novel_v4  # noqa
import search.problems_novel_v5  # noqa
import search.problems_novel_v6
import search.problems_comm_v7  # noqa
from search.profiling import profile_schedule, format_profiling_report
from search.agent_simulator_config import (
    run_profiling_agent, refine_simulator, AgentSimulator,
    _HARDWARE_MEASUREMENTS,
)
from codegen.python_wrapper import emit_python_wrapper


DEFAULT_NUM_DEVICES = 16
DEFAULT_CORES_PER_DEVICE = 2


def _extract_op_costs(agent_sim):
    """Extract per-op costs that the agent actually measured via tool calls.

    Only returns ops the Phase 1 agent profiled via measure_xla_op_overhead.
    If the agent only measured 5 ops, downstream phases only see those 5 costs.
    Unmeasured ops fall back to 29.0us in the benchmark function.
    """
    return dict(agent_sim.knowledgebase.get("measured_op_costs", {}))


def _extract_memcpy_bw(agent_sim):
    """Extract on-device memory-copy throughput for the benchmark in
    both regimes: strided (sub-region gather, e.g., narrow on
    non-leading dim) and sequential (dense permute, e.g., result of
    permute/transpose). Returns (strided_bps, seq_bps); each is 0.0
    if the agent never called measure_memory_copy_throughput, which
    disables the corresponding charge.
    """
    info = agent_sim.knowledgebase.get("memcpy_bytes_per_us", {})
    return (float(info.get("strided", 0.0)),
            float(info.get("sequential", 0.0)))


def _problem_train_scale_multiplier(problem):
    """Map from correctness-test tensor size to typical training-scale size
    for the given collective. The benchmark uses small inputs for fast
    op-count tracing, but real training tensors are much larger; this
    multiplier scales the largest observed collective tensor up so the
    NEFF compilation cost term and the HBM peak-bytes penalty in
    `benchmark_xla_candidate_generic` reflect real-training behavior.

    Values are derived from each problem's training_validation_code vs
    typical decoder-style MoE sizes (BSZ*SEQLEN*DM*K*N etc) on the
    7-node trn1.32xlarge cluster (world=224, 10B-parameter OLMoE).
    """
    return {
        # AllToAllV / Uniform A2A: MoE dispatch buffers at training scale
        # are O(BSZ * SEQLEN * TOPK * DM) per rank vs world*max_chunk in tests.
        "alltoallv":              128.0,
        "uniform_a2a":            128.0,
        # KV cache slice: ~per-rank seq * DM at training scale.
        "ring_kv":                256.0,
        # Replicated-grad AR: 10B params / 224 ranks ~ 89 MB per-rank shard
        # at bf16; cat'd over ~30 replicated grad tensors is ~2.5 GB. Test
        # uses 32 grads of 64 floats each (~8 KB cat'd). Ratio ~3.0e5. The
        # huge multiplier exposes the cat-all-grads HBM cost, while the
        # bucket-cap clamp in the cost model lets bucketed algorithms keep
        # their advertised cap regardless.
        "grad_ar":                300000.0,
        # FSDP per-layer shard at training scale is several MB; cat'd
        # over M*N_LAYERS microbatch x layer combinations is hundreds of
        # MB. Test uses ~1 KB shards. Ratio ~1e4 keeps a fully-stacked
        # FSDP candidate's NEFF compile cost realistic without hard
        # rejection.
        "fsdp_prefetch":          10000.0,
        # TP MLP partial sums at training scale are ~ BSZ * SEQLEN * DM
        # ~ few MB per layer; cat'd over M*N_LAYERS is tens of MB.
        "tp_mlp":                 10000.0,
        # PP send/recv activations at training scale are ~ BSZ * SEQLEN *
        # DM, a few MB per microbatch; the v6 "single AR over a half x M
        # buffer" pattern peaks at ~ half * M * act_bytes which is ~ 1 GB
        # at training scale. Multiplier 300 keeps that peak under the
        # 2 GB single-tensor budget while still exposing the cat-all
        # peak of any candidate that materializes a multi-microbatch
        # bundle.
        "pp_send_recv":           300.0,
        # Llama transformer-block AR: attn/mlp partials at training scale
        # are ~ BSZ * SEQLEN * DM. Test uses ~16 KB; ratio ~10x.
        "llama_block_ar":         100.0,
        # DXE: logits[T, V_local]. Modest scale-up; test already uses
        # representative-ish sizes.
        "dxe":                    10.0,
    }.get(getattr(problem, "name", ""), 1.0)


def make_send_counts(pattern, world=32, shard_size=1024):
    """Generate a send_counts_matrix[src][dst] for the given traffic pattern."""
    matrix = [[0] * world for _ in range(world)]
    if pattern == "moe":
        rng = random.Random(42)
        raw = [1.0 / (i + 1) ** 1.2 for i in range(world)]
        perm = list(range(world))
        rng.shuffle(perm)
        probs = [0.0] * world
        for i, p in enumerate(perm):
            probs[p] = raw[i]
        total_p = sum(probs)
        probs = [p / total_p for p in probs]
        cdf = []
        acc = 0.0
        for p in probs:
            acc += p
            cdf.append(acc)
        for s in range(world):
            counts = [0] * world
            for _ in range(shard_size):
                r = rng.random()
                for d in range(world):
                    if r <= cdf[d]:
                        counts[d] += 1
                        break
            matrix[s] = counts
    elif pattern == "uniform":
        for s in range(world):
            for d in range(world):
                matrix[s][d] = shard_size
    elif pattern == "skewed":
        for s in range(world):
            for d in range(world):
                matrix[s][d] = shard_size * 4 if d in (0, 1) else 128
    elif pattern == "sparse":
        random.seed(42)
        for s in range(world):
            for d in random.sample(range(world), max(1, world // 4)):
                matrix[s][d] = shard_size
    elif pattern == "random":
        random.seed(42)
        for s in range(world):
            for d in range(world):
                matrix[s][d] = random.randint(0, shard_size * 2)
    elif pattern == "increasing":
        for s in range(world):
            for d in range(world):
                matrix[s][d] = (d + 1) * (shard_size // world)
    elif pattern == "locality":
        for s in range(world):
            for d in range(world):
                dist = min(abs(s - d), world - abs(s - d))
                matrix[s][d] = max(1, shard_size // (1 + dist))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return matrix


def print_results(results, top_n=12):
    print(f"\n{'#':<3} {'Name':<35} {'Template':<18} {'Score':<9} "
          f"{'SimTime(us)':<12} {'Dispatches':<11} {'Hops':<7} {'Contention':<11} {'Steps':<6}")
    print("-" * 115)
    for i, (name, metrics) in enumerate(results[:top_n]):
        tmpl = metrics.get("template", "?")
        print(f"{i+1:<3} {name:<35} {tmpl:<18} {metrics['cost_score']:<9.3f} "
              f"{metrics['sim_time_us']:<12.1f} {metrics.get('num_dispatches','?'):<11} "
              f"{metrics.get('hop_cost',0):<7.2f} "
              f"{metrics.get('contention',0):<11.2f} {metrics.get('num_steps',0):<6}")


# ================================================================
# Phase 1: Agent Hardware Profiling (LLM builds simulator)
# ================================================================

def phase1_profiling(use_llm, llm_model, num_nodes, verbose=True):
    """
    LLM profiles hardware and builds its own cost model simulator.

    The agent:
    1. Discovers topology (device count, adjacency, link properties)
    2. Measures collective dispatch overhead, XLA op overhead, bandwidth
    3. Writes a Python cost model: estimate_latency(algo, **kwargs) -> us
    4. Validates predictions match real measurements within 20%
    5. Iterates until validated

    Returns:
        agent_sim: AgentSimulator with config + cost function
        topology: TrainiumTopology or MultiNodeTopology
    """
    print("\n" + "=" * 70)
    print("[Phase 1] Agent Hardware Profiling — LLM Builds Simulator")
    print("=" * 70)

    dispatch_overhead = 100.0
    agent_sim = AgentSimulator()

    # Phase 1 is deterministic across search back-ends (kiss / strat / cc-react):
    # run the same set of hardware probes for all, so the cost model in
    # phase 3/4/5 gets identical calibrated values regardless of who calls
    # phase1_profiling. Replaces the earlier LLM-driven tool-exploration path
    # which took 15-25 min per invocation (LLM burned turns enumerating
    # measure_* tools before converging, often timed out entirely), and gave
    # STRICTLY LESS information than the deterministic path (the LLM only
    # called a subset of tools each run). Auto-probe calls each tool once
    # with its canonical inputs so downstream sees identical calibrated values.
    print("  Deterministic phase-1 auto-probe (LLM-driven path deprecated)")
    try:
        from search.agent_simulator_config import _handle_tool_call, _HARDWARE_MEASUREMENTS
        _handle_tool_call("measure_memory_copy_throughput", {}, agent_sim)
        _handle_tool_call("measure_graph_launch_overhead", {}, agent_sim)
        for _depth in (1, 2, 4, 8, 16):
            _handle_tool_call("measure_back_to_back_amortization",
                              {"depth": _depth}, agent_sim)
        _handle_tool_call("measure_standalone_graph_cost", {}, agent_sim)
        # Seed dispatch_overhead from the depth-1 back-to-back probe.
        _b2b_hm = _HARDWARE_MEASUREMENTS.get("back_to_back_amortization_us", {}) or {}
        dispatch_overhead = float(_b2b_hm.get("dispatch_overhead_us_first_issue", 100.0))
        try:
            agent_sim.config.collective_dispatch_overhead_us = dispatch_overhead
        except Exception:
            pass
    except Exception as _e:
        print(f"  auto-probe failed: {_e}")

    # Use agent-discovered topology structure (num_devices, adjacency) but
    # apply sanity checks on bandwidth/latency — the agent often mis-derives
    # these from small-message measurements where fixed overhead dominates.
    config = agent_sim.config

    # Discover unsupported primitives via compilation test if not yet populated
    if not config.unsupported_primitives:
        from search.agent_simulator_config import _test_primitive_compilation
        for prim in ["all_gather", "reduce_scatter", "all_reduce",
                     "collective_permute", "all_to_all",
                     "cumsum", "cumprod", "sort", "argsort"]:
            try:
                result = _test_primitive_compilation(prim)
                if not result.get("compiles_on_hardware", True):
                    config.unsupported_primitives.append(prim)
            except Exception:
                pass
    _ESSENTIAL = {"all_gather", "reduce_scatter", "all_reduce", "collective_permute"}
    if _ESSENTIAL.intersection(config.unsupported_primitives):
        config.unsupported_primitives = [
            p for p in config.unsupported_primitives if p not in _ESSENTIAL
        ]
    _nd = config.num_devices if config.num_devices > 0 else DEFAULT_NUM_DEVICES
    _cpd = config.cores_per_device if config.cores_per_device > 0 else DEFAULT_CORES_PER_DEVICE
    _adj = config.device_adjacency if config.device_adjacency else None

    # Bandwidth: agent may report absurdly low values (e.g. 1.5 GB/s from
    # small-message P2P where latency dominates). Use default if < 10 GB/s.
    _bw = config.link_bandwidth_gbps if config.link_bandwidth_gbps >= 10.0 else 192.0
    _lat = config.link_latency_us if 0 < config.link_latency_us < 100 else 0.5

    # Dispatch overhead: agent may conflate collective total latency with
    # per-dispatch overhead. Use default if > 500 us (real is ~100 us).
    if dispatch_overhead > 500.0:
        print(f"  WARNING: Agent set dispatch_overhead={dispatch_overhead:.0f} us "
              f"(likely confused with collective latency). Using default 100 us.")
        dispatch_overhead = 100.0

    if num_nodes > 1:
        topology = MultiNodeTopology(
            num_nodes=num_nodes,
            neuronlink_bandwidth_GBps=_bw, neuronlink_latency_us=_lat,
            num_devices_per_node=_nd, cores_per_device=_cpd,
            device_adjacency=_adj)
    else:
        topology = TrainiumTopology(
            link_bandwidth_GBps=_bw, link_latency_us=_lat,
            num_devices=_nd, cores_per_device=_cpd,
            device_adjacency=_adj)
    topology.summary()

    unsupported_primitives = agent_sim.config.unsupported_primitives
    if unsupported_primitives:
        print(f"  Hardware constraints: unsupported primitives = {unsupported_primitives}")

    return agent_sim, topology, dispatch_overhead


# ================================================================
# Phase 2: Baseline Evaluation on Simulator → Knowledgebase
# ================================================================

def phase2_baseline_eval(topology, send_counts, dispatch_overhead,
                         agent_sim, use_llm, llm_model, num_nodes,
                         ga_generations, ga_population, sa_iters,
                         verbose=True):
    """
    Evaluate all algorithm templates on the simulator to build a knowledgebase
    of how different approaches perform.

    Includes:
    - Default configs for all templates (permute_ring, allgather_slice,
      hierarchical, pairwise, hybrid, fused_alltoall, allgather_reduce_scatter)
    - GA/SA refinement of parametric templates
    - LLM-generated schedule candidates

    If simulator predictions seem inconsistent, returns feedback for Phase 1
    refinement.

    Returns:
        all_results: ranked list of (name, metrics)
        knowledgebase: dict summarizing what was learned
        refinement_needed: str or None (feedback for Phase 1 if simulator needs fixing)
    """
    print("\n" + "=" * 70)
    print("[Phase 2] Baseline Evaluation on Simulator → Knowledgebase")
    print("=" * 70)

    world = topology.num_cores
    num_devices = topology.num_devices
    cost_model = CostModel(topology, send_counts,
                           dispatch_overhead_us=dispatch_overhead)
    all_results = []

    # --- Evaluate all templates with default params ---
    print("\n  [2a] Default configurations...")

    for sched_name, sched in [
        ("pr:default_ring", list(range(1, world))),
        ("pr:topology_aware", topology_aware_schedule(topology, world)),
        ("pr:contention_greedy", contention_aware_schedule(topology, world)),
        ("pr:traffic_adaptive", traffic_adaptive_schedule(topology, send_counts, world)),
        ("pr:butterfly", butterfly_schedule(world)),
    ]:
        params = {"schedule": sched}
        m = evaluate_template("permute_ring", params, send_counts, topology)
        m["_params"] = params
        all_results.append((sched_name, m))

    for cf in [1, 2, 4]:
        params = {"chunk_factor": cf}
        name = f"ag:chunk={cf}"
        m = evaluate_template("allgather_slice", params, send_counts, topology)
        m["_params"] = params
        all_results.append((name, m))

    hier_default = hierarchical_default_params(world, num_devices)
    m = evaluate_template("hierarchical", hier_default, send_counts, topology)
    m["_params"] = hier_default
    all_results.append(("hier:default", m))

    hier_topo = {"inter_schedule": []}
    dev_hop_costs = {}
    for d in range(1, num_devices):
        total = sum(topology.device_hops(dev, (dev + d) % num_devices)
                    for dev in range(num_devices))
        dev_hop_costs[d] = total
    hier_topo["inter_schedule"] = sorted(range(1, num_devices),
                                          key=lambda d: dev_hop_costs[d])
    m = evaluate_template("hierarchical", hier_topo, send_counts, topology)
    m["_params"] = hier_topo
    all_results.append(("hier:topo_aware", m))

    pw_params = pairwise_default_params(world)
    m = evaluate_template("pairwise", pw_params, send_counts, topology)
    m["_params"] = pw_params
    all_results.append(("pw:default", m))

    hyb_params = hybrid_default_params(topology, world)
    m = evaluate_template("hybrid_ag_perm", hyb_params, send_counts, topology)
    m["_params"] = hyb_params
    all_results.append(("hyb:default", m))

    m = evaluate_template("fused_alltoall", {}, send_counts, topology)
    m["_params"] = {}
    all_results.append(("fused:default", m))

    m = evaluate_template("allgather_reduce_scatter", {}, send_counts, topology)
    m["_params"] = {}
    all_results.append(("ag_rs:default", m))

    if num_nodes > 1:
        from search.templates import multinode_hierarchical_default_params
        mn_hier_params = multinode_hierarchical_default_params(
            world, num_devices, num_nodes)
        m = evaluate_template("multinode_hierarchical", mn_hier_params,
                              send_counts, topology)
        m["_params"] = mn_hier_params
        all_results.append(("mn_hier:default", m))

    all_results.sort(key=lambda x: x[1]["cost_score"])
    print(f"  Evaluated {len(all_results)} default configurations")
    print_results(all_results, top_n=8)

    # --- GA refinement (parallelized: 3-4 independent tracks via fork) ---
    matchings = _generate_matchings(world)
    pw_elements = list(range(world - 1))
    hyb_base = hybrid_default_params(topology, world)

    def pr_cost(sched):
        s, _ = cost_model.evaluate_template("permute_ring", {"schedule": sched})
        return s

    def hier_cost(sched):
        s, _ = cost_model.evaluate_template("hierarchical", {"inter_schedule": sched})
        return s

    def pw_cost(order):
        s, _ = cost_model.evaluate_template(
            "pairwise", {"round_order": order, "_matchings": matchings})
        return s

    def hyb_cost(sched):
        p = dict(hyb_base)
        p["permute_schedule"] = sched
        s, _ = cost_model.evaluate_template("hybrid_ag_perm", p)
        return s

    pr_seeds = [r[1]["_params"]["schedule"] for r in all_results
                if r[1].get("template") == "permute_ring"][:5]
    hier_seeds = [r[1]["_params"]["inter_schedule"] for r in all_results
                  if r[1].get("template") == "hierarchical"
                  and "inter_schedule" in r[1].get("_params", {})][:3]

    def _run_pr(q):
        try:
            r = genetic_search(pr_cost, world, ga_population, ga_generations,
                               seed_schedules=pr_seeds)
            q.put(("pr", r, None))
        except Exception as e:
            q.put(("pr", None, repr(e)))

    def _run_hier(q):
        try:
            r = genetic_search(hier_cost, num_devices, ga_population, ga_generations,
                               seed_schedules=hier_seeds)
            q.put(("hier", r, None))
        except Exception as e:
            q.put(("hier", None, repr(e)))

    def _run_pw(q):
        try:
            r = genetic_search(pw_cost, world, ga_population, ga_generations,
                               elements=pw_elements)
            q.put(("pw", r, None))
        except Exception as e:
            q.put(("pw", None, repr(e)))

    def _run_hyb(q):
        try:
            r = genetic_search(hyb_cost, world=world,
                               population_size=ga_population, generations=ga_generations,
                               seed_schedules=[hyb_base["far_distances"]],
                               elements=hyb_base["far_distances"])
            q.put(("hyb", r, None))
        except Exception as e:
            q.put(("hyb", None, repr(e)))

    tracks = [("pr", _run_pr), ("hier", _run_hier), ("pw", _run_pw)]
    if hyb_base["far_distances"]:
        tracks.append(("hyb", _run_hyb))

    print(f"\n  [2b] GA refinement ({ga_generations} gens, pop={ga_population}) "
          f"\u2014 {len(tracks)} tracks in parallel via multiprocessing")

    ctx = _mp.get_context("fork")
    q = ctx.Queue()
    procs = []
    import time as _t
    t0 = _t.time()
    for name, fn in tracks:
        p = ctx.Process(target=fn, args=(q,), name=f"ga_{name}")
        p.start()
        procs.append(p)

    results = {}
    expected = len(tracks)
    while len(results) < expected:
        name, r, err = q.get()
        elapsed = _t.time() - t0
        if err is None:
            results[name] = r
            print(f"    track {name}: done  ({elapsed:.1f}s elapsed)")
        else:
            print(f"    track {name}: FAILED ({err})  ({elapsed:.1f}s elapsed)")
            results[name] = None

    for p in procs:
        p.join()

    # Integrate results back into all_results in the original order
    if results.get("pr") is not None:
        ga_pr, ga_pr_cost, _ = results["pr"]
        m = evaluate_template("permute_ring", {"schedule": ga_pr}, send_counts, topology)
        m["_params"] = {"schedule": ga_pr}
        all_results.append(("pr:GA_refined", m))

    if results.get("hier") is not None:
        ga_hier, ga_hier_cost, _ = results["hier"]
        m = evaluate_template("hierarchical", {"inter_schedule": ga_hier},
                              send_counts, topology)
        m["_params"] = {"inter_schedule": ga_hier}
        all_results.append(("hier:GA_refined", m))

    if results.get("pw") is not None:
        ga_pw, ga_pw_cost, _ = results["pw"]
        pw_params_ga = {"round_order": ga_pw, "_matchings": matchings}
        m = evaluate_template("pairwise", pw_params_ga, send_counts, topology)
        m["_params"] = pw_params_ga
        all_results.append(("pw:GA_refined", m))

    if "hyb" in results and results["hyb"] is not None:
        ga_hyb, ga_hyb_cost, _ = results["hyb"]
        hyb_ga = dict(hyb_base)
        hyb_ga["permute_schedule"] = ga_hyb
        m = evaluate_template("hybrid_ag_perm", hyb_ga, send_counts, topology)
        m["_params"] = hyb_ga
        all_results.append(("hyb:GA_refined", m))

    # --- SA polish on best ---
    all_results.sort(key=lambda x: x[1]["cost_score"])
    best_name, best_m = all_results[0]
    best_template = best_m["template"]
    best_params = best_m["_params"]

    print(f"\n  [2c] SA polish on best ({best_name})...")

    if best_template == "permute_ring":
        sa_best, sa_cost, _ = simulated_annealing(
            pr_cost, world, best_params["schedule"], sa_iters)
        sa_params = {"schedule": sa_best}
    elif best_template == "hierarchical":
        sa_best, sa_cost, _ = simulated_annealing(
            hier_cost, num_devices, best_params["inter_schedule"], sa_iters)
        sa_params = {"inter_schedule": sa_best}
    elif best_template == "pairwise":
        sa_best, sa_cost, _ = simulated_annealing(
            pw_cost, world, best_params["round_order"], sa_iters,
            elements=pw_elements)
        sa_params = {"round_order": sa_best, "_matchings": matchings}
    else:
        sa_params = best_params
        sa_cost = best_m["cost_score"]

    m = evaluate_template(best_template, sa_params, send_counts, topology)
    m["_params"] = sa_params
    all_results.append((f"{best_template[:4]}:SA_polished", m))

    # --- Local search ---
    all_results.sort(key=lambda x: x[1]["cost_score"])
    ls_name, ls_m = all_results[0]
    ls_template = ls_m["template"]
    ls_params = ls_m["_params"]

    if ls_template == "permute_ring":
        ls_best, ls_cost, ls_rounds = local_search(pr_cost, ls_params["schedule"])
        ls_final = {"schedule": ls_best}
    elif ls_template == "hierarchical":
        ls_best, ls_cost, ls_rounds = local_search(hier_cost, ls_params["inter_schedule"])
        ls_final = {"inter_schedule": ls_best}
    elif ls_template == "pairwise":
        ls_best, ls_cost, ls_rounds = local_search(pw_cost, ls_params["round_order"])
        ls_final = {"round_order": ls_best, "_matchings": matchings}
    else:
        ls_best, ls_cost, ls_rounds = None, ls_m["cost_score"], 0
        ls_final = ls_params

    if ls_best is not None:
        m = evaluate_template(ls_template, ls_final, send_counts, topology)
        m["_params"] = ls_final
        all_results.append((f"{ls_template[:4]}:local_search", m))

    # --- LLM candidates ---
    if use_llm:
        print(f"\n  [2d] LLM schedule candidates ({llm_model})...")
        traffic_desc = {
            "uniform": "All ranks send equal amounts to all other ranks.",
            "skewed": "Ranks 0,1 (device 0) receive 4x more data (MoE hotspot).",
            "sparse": "Only 25% of rank pairs exchange data.",
            "random": "Random send counts between 0 and 2*shard_size.",
            "locality": "Nearby ranks exchange more data (spatial locality).",
            "increasing": "Linearly increasing send counts by destination rank.",
        }.get("moe", "MoE traffic with Zipf-distributed expert popularity.")

        llm_results = generate_llm_schedule(
            send_counts, f"MoE traffic. {traffic_desc}",
            model=llm_model, num_candidates=3, temperature=1.0)

        for i, (sched, reasoning) in enumerate(llm_results):
            params = {"schedule": sched}
            m = evaluate_template("permute_ring", params, send_counts, topology)
            m["_params"] = params
            all_results.append((f"llm:{llm_model}_{i}", m))

    all_results.sort(key=lambda x: x[1]["cost_score"])

    # --- Build knowledgebase ---
    knowledgebase = _build_knowledgebase(all_results, topology, cost_model)

    print(f"\n  Knowledgebase summary:")
    print(f"    Total candidates: {len(all_results)}")
    print(f"    Best template: {all_results[0][1]['template']}")
    print(f"    Best score: {all_results[0][1]['cost_score']:.3f}")
    print(f"    Best sim_time: {all_results[0][1]['sim_time_us']:.1f} us")

    # Check for simulator inconsistencies
    refinement_needed = _check_simulator_consistency(all_results, agent_sim)

    return all_results, knowledgebase, cost_model, refinement_needed


def _build_knowledgebase(results, topology, cost_model):
    """Summarize findings into a knowledgebase for downstream phases."""
    kb = {
        "num_candidates": len(results),
        "top_templates": [],
        "dispatch_dominance": False,
    }

    seen_templates = set()
    for name, m in results[:10]:
        tmpl = m["template"]
        if tmpl not in seen_templates:
            seen_templates.add(tmpl)
            kb["top_templates"].append({
                "template": tmpl,
                "best_name": name,
                "score": m["cost_score"],
                "sim_time_us": m["sim_time_us"],
                "dispatches": m.get("num_dispatches", "?"),
            })

    if kb["top_templates"]:
        best = kb["top_templates"][0]
        dispatches = best.get("dispatches", 0)
        if isinstance(dispatches, int) and dispatches <= 3:
            kb["dispatch_dominance"] = True

    return kb


def _check_simulator_consistency(results, agent_sim):
    """Check if simulator predictions are self-consistent."""
    if not agent_sim.cost_function:
        return None

    top = results[0][1] if results else None
    if top and top.get("sim_time_us", 0) < 10:
        return ("Simulator predicts unrealistically low latency "
                f"({top['sim_time_us']:.1f} us). Check dispatch overhead model.")
    return None


# ================================================================
# Phase 3: Multi-island Evolution with Simulator Feedback
# ================================================================

def phase3_evolution(topology, send_counts, cost_model, all_results,
                     knowledgebase, agent_sim, use_llm, llm_model,
                     num_nodes, ga_generations, ga_population,
                     dispatch_overhead_us=100.0, verbose=True):
    """
    Multi-island LLM-guided evolution + template evolution.

    Uses simulator feedback to guide search. Includes:
    - Island evolution (3 islands: latency, contention, hop-cost)
    - CGIS refinement (contention-guided iterative synthesis)
    - Template evolution (LLM synthesizes new algorithm code)

    If evolved candidates have unexpected simulation profiles, returns
    feedback for Phase 1 simulator refinement.

    Returns:
        all_results: updated ranked list
        refinement_needed: str or None
    """
    print("\n" + "=" * 70)
    print("[Phase 3] Multi-island Evolution with Simulator Feedback")
    print("=" * 70)

    world = topology.num_cores
    num_devices = topology.num_devices
    analyzer = ContentionAnalyzer(topology, send_counts)

    # --- Island evolution ---
    if use_llm:
        print(f"\n  [3a] Island evolution with LLM crossover...")
        island_evo = IslandEvolution(
            topology, send_counts, cost_model, analyzer, model=llm_model)

        pr_seeds = [r[1]["_params"]["schedule"] for r in all_results
                    if r[1].get("template") == "permute_ring"][:5]
        ie_pr, ie_pr_cost, _ = island_evo.evolve(
            template="permute_ring", generations=ga_generations,
            island_pop=ga_population // 2, migration_interval=25,
            llm_crossover_count=2, seed_schedules=pr_seeds, verbose=verbose)
        m = evaluate_template("permute_ring", {"schedule": ie_pr},
                              send_counts, topology)
        m["_params"] = {"schedule": ie_pr}
        all_results.append(("pr:island_evo", m))

        hier_seeds = [r[1]["_params"]["inter_schedule"] for r in all_results
                      if r[1].get("template") == "hierarchical"
                      and "inter_schedule" in r[1].get("_params", {})][:3]
        ie_hier, ie_hier_cost, _ = island_evo.evolve(
            template="hierarchical",
            elements=list(range(1, num_devices)),
            generations=ga_generations,
            island_pop=ga_population // 2, migration_interval=25,
            llm_crossover_count=2, seed_schedules=hier_seeds, verbose=verbose)
        m = evaluate_template("hierarchical", {"inter_schedule": ie_hier},
                              send_counts, topology)
        m["_params"] = {"inter_schedule": ie_hier}
        all_results.append(("hier:island_evo", m))

    # --- CGIS refinement ---
    if use_llm:
        all_results.sort(key=lambda x: x[1]["cost_score"])
        cgis_name, cgis_m = all_results[0]
        cgis_template = cgis_m["template"]
        cgis_params = cgis_m["_params"]

        print(f"\n  [3b] CGIS refinement on best ({cgis_name})...")

        cgis = IterativeRefinement(
            topology, send_counts, cost_model, analyzer, model=llm_model,
            use_profiling=True)

        if cgis_template == "permute_ring":
            cgis_sched = cgis_params["schedule"]
        elif cgis_template == "hierarchical":
            cgis_sched = cgis_params["inter_schedule"]
        else:
            cgis_sched = cgis_params.get("schedule",
                                          cgis_params.get("inter_schedule"))

        if cgis_sched:
            cgis_best, cgis_cost, _ = cgis.refine(
                cgis_sched, template=cgis_template,
                max_rounds=6, patience=3, verbose=verbose)

            if cgis_template == "permute_ring":
                cgis_final_params = {"schedule": cgis_best}
            else:
                cgis_final_params = {"inter_schedule": cgis_best}

            m = evaluate_template(cgis_template, cgis_final_params,
                                  send_counts, topology)
            m["_params"] = cgis_final_params
            all_results.append((f"{cgis_template[:4]}:CGIS_refined", m))

    # --- Template evolution (LLM-synthesized algorithm code) ---
    if use_llm:
        print(f"\n  [3c] Template evolution (LLM algorithm synthesis)...")
        unsup = agent_sim.config.unsupported_primitives if agent_sim else None
        op_costs = _extract_op_costs(agent_sim)
        te = TemplateEvolution(
            topology, send_counts, cost_model, analyzer, model=llm_model,
            unsupported_primitives=unsup, op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead_us)
        for starting in ["ag_slice_cat", "allgather_reduce_scatter",
                         "permute_ring"]:
            try:
                evo_code, evo_bench, evo_hist = te.evolve(
                    starting_template=starting, max_rounds=8, verbose=verbose)
                if evo_bench and "sim_time_us" in evo_bench:
                    sim_us = evo_bench["sim_time_us"]
                    evo_m = {
                        "template": f"evolved_{starting}",
                        "cost_score": sim_us / 100.0,
                        "sim_time_us": sim_us,
                        "num_steps": evo_bench.get("steps", 0),
                        "num_collective_permute": evo_bench.get("num_collective_permute", 0),
                        "num_all_gather": evo_bench.get("num_all_gather", 0),
                        "_params": {"evolved_code": evo_code},
                    }
                    all_results.append((f"evo:{starting}", evo_m))
            except Exception as e:
                print(f"  Template evolution ({starting}) failed: {e}")

    all_results.sort(key=lambda x: x[1]["cost_score"])

    # --- Profile top candidates ---
    print(f"\n  [3d] Profiling top candidates...")
    for name, metrics in all_results[:6]:
        try:
            prof = profile_schedule(
                metrics["template"], metrics["_params"], send_counts, topology)
            bottlenecks = prof.bottleneck_steps(3)
            top_steps = ", ".join(
                f"step {prof.step_details[i]['step']}"
                f"(d={prof.step_details[i].get('distance','?')})"
                f"={prof.step_time_us(i):.1f}us"
                for i in bottlenecks
            )
            print(f"  {name}: {prof.total_time_us:.1f}us "
                  f"(eff={prof.efficiency():.1%}) "
                  f"bottlenecks=[{top_steps}]")
            metrics["_profiling"] = format_profiling_report(prof)
        except Exception:
            pass

    # Final ranking after evolution
    print("\n  Evolution results:")
    print_results(all_results, top_n=10)

    refinement_needed = _check_evolution_consistency(all_results, knowledgebase)
    return all_results, refinement_needed


def _check_evolution_consistency(results, knowledgebase):
    """Check if evolution found something the simulator baseline missed."""
    if not results:
        return None
    best = results[0][1]
    if best.get("template", "").startswith("evolved_"):
        baseline_best = knowledgebase.get("top_templates", [{}])[0]
        if baseline_best and best["cost_score"] < baseline_best.get("score", float("inf")) * 0.5:
            return ("Evolved template is >2x better than best baseline. "
                    "Simulator may be miscalibrated for novel algorithms.")
    return None


# ================================================================
# Phase 4: Iterative Mini-benchmarking + Refining on Real HW
# ================================================================

def phase4_hardware_eval(all_results, send_counts, topology, agent_sim,
                         use_llm, llm_model, num_nodes, worker_addrs,
                         verbose=True):
    """
    Run top candidates on real hardware and compare with simulator predictions.

    If real HW latency diverges significantly from simulator predictions,
    feeds errors back to refine the simulator (Phase 1 loop-back).

    Returns:
        hw_results: list of (name, hw_latency_ms, sim_latency_us)
        agent_sim: potentially refined AgentSimulator
    """
    print("\n" + "=" * 70)
    print("[Phase 4] Iterative Mini-benchmarking on Real Hardware")
    print("=" * 70)

    world = topology.num_cores
    _master_addr = "localhost"
    if num_nodes > 1 and worker_addrs:
        import socket
        _master_addr = socket.gethostbyname(socket.gethostname())

    hw_results = []

    # Baseline
    print(f"\n  Running baseline (default_ring)...")
    hw_baseline = run_on_hardware(
        "permute_ring", {"schedule": list(range(1, world))}, send_counts,
        num_nodes=num_nodes, master_addr=_master_addr,
        worker_addrs=worker_addrs)
    if hw_baseline and hw_baseline.get("hw_latency_ms"):
        hw_results.append(("baseline:default_ring",
                           hw_baseline["hw_latency_ms"], None))
        print(f"  Baseline: {hw_baseline['hw_latency_ms']:.3f} ms")

    # Top unique templates
    seen_templates = set()
    candidates_run = 0
    max_hw_candidates = 6

    for name, metrics in all_results[:12]:
        if candidates_run >= max_hw_candidates:
            break
        tmpl = metrics["template"]
        if tmpl in seen_templates:
            continue
        seen_templates.add(tmpl)

        params = metrics["_params"]
        sim_us = metrics.get("sim_time_us", 0)

        print(f"  Running {name} ({tmpl}, sim={sim_us:.1f} us)...")
        hw = run_on_hardware(
            tmpl, params, send_counts,
            num_nodes=num_nodes, master_addr=_master_addr,
            worker_addrs=worker_addrs)

        if hw and hw.get("hw_latency_ms"):
            hw_ms = hw["hw_latency_ms"]
            print(f"    HW: {hw_ms:.3f} ms | Sim: {sim_us:.1f} us | "
                  f"Ratio: {hw_ms * 1000 / max(sim_us, 0.1):.2f}x")
            # Training validation gate
            alltoallv_prob = PROBLEMS.get("alltoallv")
            evolved_code = params.get("evolved_code", params.get("builtin_code", ""))
            if alltoallv_prob and alltoallv_prob.training_validation_code and evolved_code:
                print(f"    Running training validation (10 steps, bf16)...")
                tv = _run_training_validation(
                    alltoallv_prob, evolved_code,
                    topology.num_cores, topology.num_devices,
                    num_nodes, _master_addr, worker_addrs)
                if tv.get("passed"):
                    hw_results.append((name, hw_ms, sim_us))
                    print(f"    Training validation: PASSED")
                else:
                    err_msg = tv.get('error', 'unknown')[:200]
                    print(f"    Training validation: FAILED ({err_msg})")
                    if use_llm:
                        fixed = _training_failure_recovery(
                            alltoallv_prob, evolved_code, tv,
                            topology, num_nodes, _master_addr,
                            worker_addrs, llm_model, verbose)
                        if fixed:
                            fixed_code, fixed_hw_ms = fixed
                            hw_results.append(
                                (name + ":tv_fix", fixed_hw_ms, sim_us))
                            params["evolved_code"] = fixed_code
                            print(f"    Recovery: PASSED ({fixed_hw_ms:.3f} ms)")
            else:
                hw_results.append((name, hw_ms, sim_us))
            candidates_run += 1
        else:
            err = hw.get("error", "unknown") if hw else "failed"
            print(f"    FAILED: {err}")
            candidates_run += 1

    # Check sim vs HW correlation and refine if needed
    if hw_results and use_llm:
        error_feedback = _analyze_hw_sim_gap(hw_results)
        if error_feedback:
            print(f"\n  Simulator-HW gap detected, refining...")
            agent_sim = refine_simulator(
                agent_sim, error_feedback, model=llm_model,
                max_turns=5, verbose=verbose)

    # Print comparison table
    if hw_results:
        print(f"\n  {'Algorithm':<35} {'HW (ms)':<10} {'Sim (us)':<10} {'Ratio':<8}")
        print("  " + "-" * 63)
        for name, hw_ms, sim_us in hw_results:
            sim_str = f"{sim_us:.1f}" if sim_us else "N/A"
            ratio = f"{hw_ms * 1000 / sim_us:.2f}x" if sim_us else "N/A"
            print(f"  {name:<35} {hw_ms:<10.3f} {sim_str:<10} {ratio:<8}")

    return hw_results, agent_sim


def _analyze_hw_sim_gap(hw_results):
    """Check if simulator predictions correlate with HW measurements."""
    pairs = [(hw_ms, sim_us) for _, hw_ms, sim_us in hw_results
             if sim_us and sim_us > 0]
    if len(pairs) < 2:
        return None

    ratios = [hw_ms * 1000 / sim_us for hw_ms, sim_us in pairs]
    avg_ratio = sum(ratios) / len(ratios)
    max_deviation = max(abs(r - avg_ratio) / avg_ratio for r in ratios)

    if max_deviation > 0.5:
        return (f"Simulator-to-HW ratios vary widely: {[f'{r:.2f}x' for r in ratios]}. "
                f"Mean ratio={avg_ratio:.2f}x, max deviation={max_deviation:.0%}. "
                f"The cost model may be weighting dispatch vs bandwidth incorrectly.")
    return None


def _run_generic_on_hardware(problem, evolved_code, world, num_devices,
                              num_nodes, master_addr, worker_addrs, timeout=900):
    """Run a generic collective candidate on real hardware."""
    import os
    import subprocess
    import tempfile
    import re as _re

    NEURON_VENV = os.environ.get(
        "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
    MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

    # Per-problem benchmark body
    if problem.name == "uniform_a2a":
        bench = """
    chunk_size = 1024
    x = torch.randn(world * chunk_size, device=device, dtype=torch.float32, requires_grad=True)
    out = EVOLVED_FN(x, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * chunk_size, device=device, dtype=torch.float32, requires_grad=True)
        out = EVOLVED_FN(x_i, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
"""
    elif problem.name == "ring_kv":
        bench = """
    k = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
    outs = EVOLVED_FN([k, v], rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    sum(o.sum() for o in outs).backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        ki = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
        vi = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
        outs = EVOLVED_FN([ki, vi], rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        sum(o.sum() for o in outs).backward()
        xla.step()
"""
    elif problem.name == "alltoallv":
        bench = """
    max_chunk = 1024
    send_counts = [max_chunk] * world
    recv_counts = [max_chunk] * world
    x = torch.randn(world * max_chunk, device=device, dtype=torch.float32, requires_grad=True)
    out = EVOLVED_FN(x, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * max_chunk, device=device, dtype=torch.float32, requires_grad=True)
        out = EVOLVED_FN(x_i, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
"""
    elif problem.name == "grad_ar":
        bench = """
    N_PARAMS = 16
    SZ = 4096
    grads = [torch.randn(SZ, device=device, dtype=torch.float32) for _ in range(N_PARAMS)]
    out = EVOLVED_FN(grads, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    [o.sum() for o in out]
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        gs = [torch.randn(SZ, device=device, dtype=torch.float32) for _ in range(N_PARAMS)]
        out = EVOLVED_FN(gs, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        [o.sum() for o in out]
        xla.step()
"""
    elif problem.name == "dxe":
        bench = """
    T = 64
    V_local = 16
    V_total = V_local * world
    logits = torch.randn(T, V_local, device=device, dtype=torch.float32)
    targets = torch.randint(0, V_total, (T,), device=device, dtype=torch.long)
    out = EVOLVED_FN(logits, targets, V_local, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    _ = out.item() if out.dim() == 0 else out.sum()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        li = torch.randn(T, V_local, device=device, dtype=torch.float32)
        ti = torch.randint(0, V_total, (T,), device=device, dtype=torch.long)
        out = EVOLVED_FN(li, ti, V_local, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        _ = out.item() if out.dim() == 0 else out.sum()
        xla.step()
"""
    else:
        return {"error": f"No HW benchmark template for {problem.name}"}

    bench = bench.replace("EVOLVED_FN", problem.evolved_fn_name)

    code = (
        "#!/usr/bin/env python3\n"
        "import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm\n"
        "import torch_xla.runtime as xr, torch.distributed as dist\n"
        "\n"
        + evolved_code.strip() + "\n"
        "\n"
        "def main():\n"
        "    device = xla.device()\n"
        "    if not dist.is_initialized():\n"
        "        dist.init_process_group('xla', init_method='xla://')\n"
        "    world = xr.world_size()\n"
        "    rank = xr.global_ordinal()\n"
        "    num_devices = world // 2\n"
        "    cpd = 2\n"
        f"    num_nodes = {num_nodes}\n"
        + bench
        + "    xm.wait_device_ops()\n"
        "    end = time.time()\n"
        "    if rank == 0:\n"
        "        print(f'latency: {(end-start)/iters*1000:.3f} ms')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/home/ubuntu",
                                     delete=False, prefix="bench_") as f:
        f.write(code)
        script_path = f.name

    try:
        torchrun_bin = os.path.join(NEURON_VENV, "bin", "torchrun")
        nproc = world // max(num_nodes, 1)
        if num_nodes > 1:
            cmd = [
                torchrun_bin,
                f"--nnodes={num_nodes}",
                f"--nproc_per_node={nproc}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_addr}:{MASTER_PORT}",
                script_path,
            ]
        else:
            cmd = [torchrun_bin, f"--nproc_per_node={nproc}", script_path]

        # Augment PATH with the Neuron venv's bin so subprocesses can find
        # libneuronpjrt-path (Neuron initialization requires it). The default
        # inherited PATH doesn't include the venv bin even though torchrun is
        # invoked by absolute path.
        env = dict(os.environ)
        venv_bin = os.path.join(NEURON_VENV, "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        if num_nodes > 1 and worker_addrs:
            from search.evaluate_algo import _run_multinode_hw
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu", env=env)
            output = result.stdout + result.stderr

        match = _re.search(r"latency:\s*([\d.]+)\s*ms", output)
        if match:
            return {
                "hw_latency_ms": float(match.group(1)),
                "output": output[-500:],
            }
        return {"hw_latency_ms": None,
                "error": ("Could not parse latency: "
                          + output[-300:].replace("\n", " | ")),
                "output": output[-1000:]}
    except subprocess.TimeoutExpired:
        return {"hw_latency_ms": None, "error": "timeout"}
    except Exception as e:
        return {"hw_latency_ms": None, "error": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _run_small_shape_microbench(problem, evolved_code, world, num_devices,
                                  num_nodes, master_addr, worker_addrs,
                                  timeout=300):
    """Run the candidate on h7-style small-shape input (mc=20 instead of
    Phase-4's mc=1024). Used by no-sim Phase-5 LLM-judge: the small-shape
    microbench is the closest analog of what the LLM would have available
    if it ran h7_bench-style probes itself, and gives a per-call latency
    that's closer to the Table-2 7-node-bench column (where AG+RS at
    1.89 ms beats ag_slice_cat at 3.04 ms) rather than the training-shape
    Phase-4 microbench (where dispatch-amortising patterns dominate)."""
    import os, subprocess, tempfile
    import re as _re

    NEURON_VENV = os.environ.get(
        "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
    # Each candidate gets its own port to avoid rendezvous collisions
    # with prior candidates that haven't fully torn down their c10d store.
    import threading as _threading
    if not hasattr(_run_small_shape_microbench, "_port_counter_lock"):
        _run_small_shape_microbench._port_counter_lock = _threading.Lock()
        _run_small_shape_microbench._port_counter = 0
    with _run_small_shape_microbench._port_counter_lock:
        _run_small_shape_microbench._port_counter += 1
        port_offset = _run_small_shape_microbench._port_counter
    MASTER_PORT = str(int(os.environ.get("MASTER_PORT_SS", "29677")) + port_offset)

    if problem.name == "uniform_a2a":
        bench = """
    chunk_size = 20
    x = torch.randn(world * chunk_size, device=device, dtype=torch.bfloat16, requires_grad=False).contiguous()
    xm.mark_step()
    out = EVOLVED_FN(x, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    xla.step()
    iters = 30
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * chunk_size, device=device, dtype=torch.bfloat16, requires_grad=False).contiguous()
        out = EVOLVED_FN(x_i, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        xla.step()
"""
    elif problem.name == "alltoallv":
        bench = """
    max_chunk = 20
    send_counts = [max_chunk] * world
    recv_counts = [max_chunk] * world
    x = torch.randn(world * max_chunk, device=device, dtype=torch.bfloat16, requires_grad=False).contiguous()
    xm.mark_step()
    out = EVOLVED_FN(x, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    xla.step()
    iters = 30
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * max_chunk, device=device, dtype=torch.bfloat16, requires_grad=False).contiguous()
        out = EVOLVED_FN(x_i, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        xla.step()
"""
    elif problem.name == "dxe":
        bench = """
    V_local = max(1, 32256 // world)
    V_total = V_local * world
    T = 64
    logits = torch.randn(T, V_local, device=device, dtype=torch.float32)
    targets = torch.randint(0, V_total, (T,), device=device, dtype=torch.long)
    xm.mark_step()
    out = EVOLVED_FN(logits, targets, V_local, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    _ = out.item() if out.dim() == 0 else out.sum()
    xla.step()
    iters = 30
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        li = torch.randn(T, V_local, device=device, dtype=torch.float32)
        ti = torch.randint(0, V_total, (T,), device=device, dtype=torch.long)
        out = EVOLVED_FN(li, ti, V_local, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        _ = out.item() if out.dim() == 0 else out.sum()
        xla.step()
"""
    else:
        return {"error": f"No small-shape bench template for {problem.name}; "
                          "falling back to training-shape Phase-4 hw."}

    bench = bench.replace("EVOLVED_FN", problem.evolved_fn_name)
    code = (
        "#!/usr/bin/env python3\n"
        "import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm\n"
        "import torch_xla.runtime as xr, torch.distributed as dist\n"
        "\n"
        + evolved_code.strip() + "\n"
        "\n"
        "def main():\n"
        "    device = xla.device()\n"
        "    if not dist.is_initialized():\n"
        "        dist.init_process_group('xla', init_method='xla://')\n"
        "    world = xr.world_size()\n"
        "    rank = xr.global_ordinal()\n"
        "    num_devices = world // 2\n"
        "    cpd = 2\n"
        f"    num_nodes = {num_nodes}\n"
        + bench
        + "    xm.wait_device_ops()\n"
        "    end = time.time()\n"
        "    if rank == 0:\n"
        "        print(f'latency: {(end-start)/iters*1000:.3f} ms')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/home/ubuntu",
                                     delete=False, prefix="ss_bench_") as f:
        f.write(code)
        script_path = f.name
    try:
        torchrun_bin = os.path.join(NEURON_VENV, "bin", "torchrun")
        nproc = world // max(num_nodes, 1)
        cmd = [
            torchrun_bin,
            f"--nnodes={num_nodes}",
            f"--nproc_per_node={nproc}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{MASTER_PORT}",
            script_path,
        ]
        env = dict(os.environ)
        venv_bin = os.path.join(NEURON_VENV, "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        if num_nodes > 1 and worker_addrs:
            from search.evaluate_algo import _run_multinode_hw
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu", env=env)
            output = result.stdout + result.stderr
        match = _re.search(r"latency:\s*([\d.]+)\s*ms", output)
        if match:
            return {"hw_latency_ms": float(match.group(1)), "output": output[-500:]}
        return {"hw_latency_ms": None,
                "error": ("Could not parse small-shape latency: "
                          + output[-300:].replace("\n", " | "))}
    except subprocess.TimeoutExpired:
        return {"hw_latency_ms": None, "error": "small-shape timeout"}
    except Exception as e:
        return {"hw_latency_ms": None, "error": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _run_training_validation(problem, evolved_code, world, num_devices,
                             num_nodes, master_addr, worker_addrs, timeout=600):
    """Run MoE-realistic training validation to verify the evolved code works
    inside torch.autograd.Function with mark_step barriers, double-call pattern
    (dispatch + combine per layer), bf16 dtypes, and a realistic XLA graph.

    The model uses 8 layers with bf16 weights and 10 training steps to stress
    the XLA compiler and catch issues that a simple 3-step fp32 test misses.

    Returns dict with 'passed' (bool), 'error' (str), 'output' (str).
    """
    import os
    import subprocess
    import tempfile

    if not problem.training_validation_code:
        return {"passed": True, "skipped": True}

    NEURON_VENV = os.environ.get(
        "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
    MASTER_PORT = os.environ.get("MASTER_PORT_TV", "29599")

    runtime_code = _emit_collective_runtime(
        problem, evolved_code, world, num_devices, num_nodes)

    parts = []
    parts.append("#!/usr/bin/env python3")
    parts.append("import os, time, torch, torch.nn as nn")
    parts.append("os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')")
    parts.append("import torch_xla as xla")
    parts.append("import torch_xla.core.xla_model as xm")
    parts.append("import torch_xla.runtime as xr")
    parts.append("import torch.distributed as dist")
    parts.append("")
    parts.append("# ---- Inlined runtime module ----")
    parts.append(runtime_code)
    parts.append("")
    # R10 TV-device-module-level fix: make `device` available to inlined
    # problem TV code (some problems materialize tensors at module level).
    parts.append("# ---- Module-level device handle for inlined problem TV code ----")
    parts.append("import torch.distributed as _tv_dist")
    parts.append("if not _tv_dist.is_initialized():")
    parts.append("    _tv_dist.init_process_group('xla', init_method='xla://')")
    parts.append("device = xla.device()")
    parts.append("")
    parts.append("# ---- Problem-specific autograd wrapper ----")
    parts.append(f"world = {world}")
    parts.append(problem.training_validation_code)
    parts.append("")
    parts.append("# ---- MoE-realistic multi-layer model ----")
    parts.append("N_LAYERS = 8")
    parts.append("N_STEPS = 10")
    parts.append("")
    parts.append("class _Layer(nn.Module):")
    parts.append("    def __init__(self, dim):")
    parts.append("        super().__init__()")
    parts.append("        self.w1 = nn.Linear(dim, dim, bias=False)")
    parts.append("        self.w2 = nn.Linear(dim, dim, bias=False)")
    parts.append("")
    parts.append("    def forward(self, x):")
    parts.append("        h = self.w1(x)")
    parts.append("        flat = h.reshape(-1)")
    parts.append("        n = flat.numel()")
    parts.append("        if n < INPUT_SIZE:")
    parts.append("            flat = torch.nn.functional.pad(flat, (0, INPUT_SIZE - n))")
    parts.append("        else:")
    parts.append("            flat = flat[:INPUT_SIZE]")
    parts.append("        out = _CollectiveOp.apply(flat)")
    parts.append("        out_n = out.numel()")
    parts.append("        if out_n >= n:")
    parts.append("            return self.w2(out[:n].reshape(h.shape))")
    parts.append("        pad_out = torch.nn.functional.pad(out, (0, n - out_n))")
    parts.append("        return self.w2(pad_out.reshape(h.shape))")
    parts.append("")
    parts.append("class _Model(nn.Module):")
    parts.append("    def __init__(self, dim, n_layers):")
    parts.append("        super().__init__()")
    parts.append("        self.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])")
    parts.append("        self.head = nn.Linear(dim, 1, bias=False)")
    parts.append("")
    parts.append("    def forward(self, x):")
    parts.append("        for layer in self.layers:")
    parts.append("            x = x + layer(x)")
    parts.append("        return self.head(x).sum()")
    parts.append("")
    parts.append("def main():")
    parts.append("    # device + dist init done at module level above")
    parts.append("    rank = xr.global_ordinal()")
    parts.append(f"    init_{problem.name}()")
    parts.append("")
    parts.append("    DIM = 1024")
    parts.append("    model = _Model(DIM, N_LAYERS).to(torch.bfloat16).to(device)")
    parts.append("    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)")
    parts.append("")
    parts.append("    # Warmup (NEFF compile + first-step transients) so the")
    parts.append("    # measured steps reflect steady-state per-step latency.")
    parts.append("    WARMUP = 3")
    parts.append("    step_us = []")
    parts.append("    for step in range(N_STEPS):")
    parts.append("        if step == WARMUP:")
    parts.append("            xm.wait_device_ops()")
    parts.append("            xm.rendezvous('tv_pre_timed')")
    parts.append("            t0 = time.time()")
    parts.append("        x = torch.randn(8, DIM, device=device, dtype=torch.bfloat16)")
    parts.append("        loss = model(x)")
    parts.append("        loss.backward()")
    parts.append("        optimizer.step()")
    parts.append("        optimizer.zero_grad()")
    parts.append("        xm.mark_step()")
    parts.append("        if rank == 0 and step % 5 == 0:")
    parts.append("            print(f'step {step} loss={loss.item():.4f}')")
    parts.append("")
    parts.append("    xm.wait_device_ops()")
    parts.append("    timed = time.time() - t0")
    parts.append("    timed_steps = N_STEPS - WARMUP")
    parts.append("    avg_step_us = (timed / max(timed_steps, 1)) * 1e6")
    parts.append("    if rank == 0:")
    parts.append("        print(f'TRAINING_VALIDATION_AVG_STEP_US={avg_step_us:.1f}')")
    parts.append("        print('TRAINING_VALIDATION_PASSED')")
    parts.append("")
    parts.append("if __name__ == '__main__':")
    parts.append("    main()")

    script = "\n".join(parts)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/home/ubuntu",
                                     delete=False, prefix="tv_") as f:
        f.write(script)
        script_path = f.name

    try:
        torchrun_bin = os.path.join(NEURON_VENV, "bin", "torchrun")
        nproc = world // max(num_nodes, 1)
        if num_nodes > 1:
            cmd = [
                torchrun_bin,
                f"--nnodes={num_nodes}",
                f"--nproc_per_node={nproc}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_addr}:{MASTER_PORT}",
                script_path,
            ]
        else:
            cmd = [torchrun_bin, f"--nproc_per_node={nproc}", script_path]

        # Same PATH augmentation so torchrun subprocesses find libneuronpjrt-path.
        env = dict(os.environ)
        venv_bin = os.path.join(NEURON_VENV, "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        if num_nodes > 1 and worker_addrs:
            from search.evaluate_algo import _run_multinode_hw
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu", env=env)
            output = result.stdout + result.stderr

        if "TRAINING_VALIDATION_PASSED" in output:
            # Extract per-step training latency if present. This is a
            # more representative ranking signal than the 20-iter
            # isolated-call HW microbench when the per-step framework
            # cost differs across candidates (e.g., NEFF cache eviction
            # under NEURON_NUM_RECENT_MODELS_TO_KEEP=1).
            avg_step_us = None
            import re as _re
            m = _re.search(
                r'TRAINING_VALIDATION_AVG_STEP_US=([\d.]+)', output)
            if m:
                try:
                    avg_step_us = float(m.group(1))
                except ValueError:
                    avg_step_us = None
            return {"passed": True, "avg_step_us": avg_step_us,
                    "output": output[-500:]}
        return {"passed": False,
                "error": "training step did not complete",
                "output": output[-1500:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _training_failure_recovery(problem, evolved_code, tv_result,
                               topology, num_nodes, master_addr,
                               worker_addrs, llm_model, verbose,
                               max_attempts=2):
    """Attempt to fix evolved code that fails training validation.

    Feeds the training error back to the LLM and asks for a fix that preserves
    the algorithm structure while addressing the specific failure.

    Returns (fixed_code, hw_latency_ms) on success, None on failure.
    """
    from search.generate_algo import _invoke_bedrock

    error_output = tv_result.get("output", "")[-800:]
    error_msg = tv_result.get("error", "unknown")

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"    Recovery attempt {attempt}/{max_attempts}...")

        prompt = f"""Your evolved collective communication function passed correctness tests and
micro-benchmarks, but FAILED during real MoE training on Trainium hardware.

## The failure
Error: {error_msg}

Output (last 800 chars):
```
{error_output}
```

## Your code that failed
```python
{evolved_code}
```

## Common training failure causes
1. **Hardcoded dtype**: Using torch.float32 instead of input_tensor.dtype — training uses bf16
2. **Non-contiguous gradient**: backward pass receives g.contiguous() but intermediate ops may
   break contiguity assumptions
3. **XLA graph compilation**: Some op patterns compile for forward but fail for backward
4. **Shape assumptions**: forward and backward may see different tensor sizes

## Your task
Fix the code so it survives 10 steps of bf16 MoE training with autograd backward passes.
Keep the same algorithmic approach — just fix the training compatibility issue.
- Use input_tensor.dtype everywhere (never hardcode float32)
- Ensure all created tensors use input_tensor.device
- Keep all index math in Python (ints/lists, not device tensors)

Return ONLY the fixed function in a ```python block. Same signature as the original."""

        try:
            response = _invoke_bedrock(prompt, model=llm_model,
                                       temperature=0.3, max_tokens=4000)
        except Exception as e:
            if verbose:
                print(f"    LLM error: {e}")
            continue

        import re
        patterns = [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]
        fixed_code = None
        for pat in patterns:
            matches = re.findall(pat, response, re.DOTALL)
            for match in matches:
                if f"def {problem.evolved_fn_name}" in match:
                    fixed_code = match.strip()
                    break
            if fixed_code:
                break

        if not fixed_code:
            if verbose:
                print(f"    Could not extract fixed code from LLM response")
            continue

        # Re-run training validation with the fix
        tv2 = _run_training_validation(
            problem, fixed_code,
            topology.num_cores, topology.num_devices,
            num_nodes, master_addr, worker_addrs)

        if tv2.get("passed"):
            hw = _run_generic_on_hardware(
                problem, fixed_code,
                topology.num_cores, topology.num_devices,
                num_nodes, master_addr, worker_addrs)
            if hw and hw.get("hw_latency_ms"):
                return fixed_code, hw["hw_latency_ms"]
            return fixed_code, 0.0
        else:
            error_output = tv2.get("output", "")[-800:]
            error_msg = tv2.get("error", "unknown")
            evolved_code = fixed_code
            if verbose:
                print(f"    Still failing: {error_msg[:100]}")

    return None


# ----------------------------------------------------------------
# Training-shape fixture gate (R20)
#
# Many candidate runtimes pass the simulator and even the existing
# uniform-shape HW microbench but crash when called with the actual
# heterogeneous tensor shapes that real training uses (e.g. OLMoE
# rep_params mix 1D RMSNorm weights with 2D QKV projections; Llama
# amp3 pp_send_recv passes a stage-0/stage-1 activation pair where
# only one side is materialized). We pre-screen each top-K candidate
# by calling its evolved fn once with a realistic training-shape
# fixture on CPU, using a stub `xm` that bypasses collectives.
# A candidate that raises ANY exception is disqualified BEFORE we
# pay the cost of a multi-node HW microbench or training-validation
# run, and is excluded from Phase-5 winner selection.
#
# Shapes are sourced from the training scripts themselves so we
# don't hardcode them here:
#  - grad_ar      : Model from training/train_olmoe10b.py rep_params
#  - tp_mlp, fsdp_prefetch, pp_send_recv, llama_block_ar : the
#    DM/HID/B/S/N_MB constants from experiments/model_extension/
#    train_llama_e2e_amp3.py
#  - alltoallv, uniform_a2a, ring_kv, dxe : OLMoE constants from
#    training/train_olmoe10b.py
# ----------------------------------------------------------------

class _StubXM:
    """Drop-in replacement for torch_xla.core.xla_model that runs on CPU.

    Returns inputs unchanged for collectives so the shape-checking
    logic in the candidate code (tensor ops feeding the collective)
    raises real Python exceptions on shape mismatches before the
    stubbed collective is invoked.
    """
    REDUCE_SUM = "sum"
    REDUCE_MAX = "max"
    REDUCE_MIN = "min"

    @staticmethod
    def all_reduce(op, x, groups=None, scale=1.0):
        return x

    @staticmethod
    def all_gather(x, dim=0, groups=None):
        # Stub-but-shape-faithful: emulate a real world_size=224
        # all_gather by replicating input ws× along dim. This lets
        # AG-then-view code (e.g. allgather_reduce_scatter) pass the
        # shape gate; conservative all_gather-no-op behavior left
        # candidates that view-back-to-(ws,...) failing artificially.
        import torch as _t
        return _t.cat([x] * 224, dim=dim)

    @staticmethod
    def reduce_scatter(op, x, scale=1.0, scatter_dim=0,
                       shard_count=1, groups=None):
        # Stub-but-shape-faithful: inverse of all_gather — slice the
        # leading dim by shard_count (taking rank-0's slice).
        if shard_count > 1 and x.shape[scatter_dim] % shard_count == 0:
            chunk = x.shape[scatter_dim] // shard_count
            slc = [slice(None)] * x.ndim
            slc[scatter_dim] = slice(0, chunk)
            return x[tuple(slc)]
        return x

    @staticmethod
    def all_to_all(x, split_dimension, concat_dimension,
                   split_count, groups=None):
        return x

    @staticmethod
    def mark_step():
        return None

    @staticmethod
    def wait_device_ops():
        return None

    @staticmethod
    def rendezvous(*args, **kwargs):
        return None


def _build_training_shape_fixture(problem_name):
    """Return (args, kwargs) to call the candidate's evolved fn with
    on CPU, using the same tensor shapes the deployed runtime would
    see in real training. Shapes are derived from the training
    scripts (no hardcoded values that aren't in those scripts).

    Returns None if no fixture is registered for this problem (the
    gate is then skipped, preserving existing behavior).
    """
    import importlib.util
    import torch as _torch

    REPO = "/home/ubuntu/agentic-collective-communication"

    def _load(path, mod_name):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        m = importlib.util.module_from_spec(spec)
        # Some training scripts do work at import time that requires
        # neuron; we don't actually execute them — we just want the
        # constants. Read constants by parsing instead.
        return spec, m

    def _read_consts(path, names):
        """Parse top-level NAME = <int literal> from a python file."""
        import ast
        with open(path) as f:
            tree = ast.parse(f.read())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    try:
                        out[tgt.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
        return out

    # World/topology stays at the deployed scale (7-node = 224 ranks).
    rank = 0
    world_size = 224
    num_devices = 112
    cores_per_device = 2
    num_nodes = 7

    if problem_name == "grad_ar":
        # Replicated-param shapes from a freshly-instantiated OLMoE Model.
        # We instantiate the Model on CPU (cheap) and pull rep_params
        # (the same set of tensors train_olmoe10b.py registers for AR).
        c = _read_consts(
            f"{REPO}/training/train_olmoe10b.py",
            {"VOCAB", "DM", "LAYERS", "HEADS"})
        VOCAB = c.get("VOCAB", 32256)
        DM = c.get("DM", 2048)
        LAYERS = c.get("LAYERS", 8)
        # Build the realistic mix: embedding + per-layer RMSNorm (1D),
        # QKV (concat 2D), router, LM head — matching the shapes the
        # OLMoE Model exposes via list(model.parameters()) filtered by
        # the "not sharded" mask the trainer uses for rep_params.
        rep_grads = []
        # Embedding (VOCAB, DM)
        rep_grads.append(_torch.randn(VOCAB, DM, dtype=_torch.bfloat16))
        # Per-layer params
        for _ in range(LAYERS):
            rep_grads.append(_torch.randn(DM, dtype=_torch.bfloat16))  # RMSNorm1
            rep_grads.append(_torch.randn(DM, dtype=_torch.bfloat16))  # RMSNorm2
            # QKV fused: (3 * DM, DM) — heterogeneous vs 1D above
            rep_grads.append(_torch.randn(3 * DM, DM, dtype=_torch.bfloat16))
            # Out proj
            rep_grads.append(_torch.randn(DM, DM, dtype=_torch.bfloat16))
            # Router gate
            rep_grads.append(_torch.randn(world_size, DM, dtype=_torch.bfloat16))
        # Final RMSNorm
        rep_grads.append(_torch.randn(DM, dtype=_torch.bfloat16))
        args = (rep_grads, rank, world_size, num_devices,
                cores_per_device, _StubXM, _torch)
        return args, {"num_nodes": num_nodes}

    if problem_name in ("tp_mlp", "fsdp_prefetch",
                        "llama_block_ar", "pp_send_recv"):
        c = _read_consts(
            f"{REPO}/experiments/model_extension/train_llama_e2e_amp3.py",
            {"DM", "HID", "N_LAYERS_PER_STAGE", "N_MB", "B", "S", "VOCAB"})
        DM = c.get("DM", 2048)
        HID = c.get("HID", 5376)
        N_LAYERS = c.get("N_LAYERS_PER_STAGE", 1)
        N_MB = c.get("N_MB", 4)
        B = c.get("B", 1)
        S = c.get("S", 256)
        if problem_name == "tp_mlp":
            partials = [
                [_torch.randn(B, S, DM, dtype=_torch.bfloat16)
                 for _ in range(N_LAYERS)]
                for _ in range(N_MB)
            ]
            args = (partials, N_MB, N_LAYERS, rank, world_size,
                    num_devices, cores_per_device, _StubXM, _torch)
            return args, {"num_nodes": num_nodes}
        if problem_name == "pp_send_recv":
            half = world_size // 2
            # In real Llama-amp3 pp_send_recv only stage-0 has the
            # full-shape activation; stage-1 ranks pass placeholders
            # (often shaped differently than the source). Provide the
            # source-stage shape; the agent code must not assume all
            # entries in `activations` have identical leading dims.
            activations = [_torch.randn(B, S, DM, dtype=_torch.bfloat16)
                           for _ in range(N_MB)]
            args = (activations, 0, half, N_MB, rank, world_size,
                    num_devices, cores_per_device, _StubXM, _torch)
            return args, {"num_nodes": num_nodes}
        if problem_name == "fsdp_prefetch":
            # Sharded weight shards (DM, HID//ws) and (HID//ws, DM)
            shard_hid = max(1, HID // world_size)
            shards = [
                _torch.randn(DM, shard_hid, dtype=_torch.bfloat16)
                for _ in range(N_LAYERS)
            ]
            args = (shards, N_LAYERS, rank, world_size,
                    num_devices, cores_per_device, _StubXM, _torch)
            return args, {"num_nodes": num_nodes}
        if problem_name == "llama_block_ar":
            attn_out = _torch.randn(B, S, DM, dtype=_torch.bfloat16)
            mlp_out = _torch.randn(B, S, DM, dtype=_torch.bfloat16)
            args = (attn_out, mlp_out, rank, world_size,
                    num_devices, cores_per_device, _StubXM, _torch)
            return args, {"num_nodes": num_nodes}

    if problem_name == "ring_kv":
        c = _read_consts(
            f"{REPO}/training/train_olmoe10b.py", {"DM"})
        DM = c.get("DM", 2048)
        k = _torch.randn(DM, dtype=_torch.bfloat16)
        v = _torch.randn(DM, dtype=_torch.bfloat16)
        args = ([k, v], rank, world_size,
                num_devices, cores_per_device, _StubXM, _torch)
        return args, {"num_nodes": num_nodes}

    if problem_name == "uniform_a2a":
        c = _read_consts(
            f"{REPO}/training/train_olmoe10b.py", {"DM"})
        DM = c.get("DM", 2048)
        chunk_size = DM
        x = _torch.randn(world_size * chunk_size, dtype=_torch.bfloat16)
        args = (x, chunk_size, rank, world_size,
                num_devices, cores_per_device, _StubXM, _torch)
        return args, {"num_nodes": num_nodes}

    if problem_name == "alltoallv":
        c = _read_consts(
            f"{REPO}/training/train_olmoe10b.py", {"DM"})
        DM = c.get("DM", 2048)
        max_chunk = DM
        send_counts = [max_chunk] * world_size
        recv_counts = [max_chunk] * world_size
        x = _torch.randn(world_size * max_chunk, dtype=_torch.bfloat16)
        args = (x, send_counts, recv_counts, max_chunk, rank, world_size,
                num_devices, cores_per_device, _StubXM, _torch)
        return args, {"num_nodes": num_nodes}

    if problem_name == "dxe":
        c = _read_consts(
            f"{REPO}/training/train_olmoe10b.py", {"VOCAB"})
        VOCAB = c.get("VOCAB", 32256)
        V_local = VOCAB // world_size
        T = 64
        logits = _torch.randn(T, V_local, dtype=_torch.float32)
        targets = _torch.randint(0, VOCAB, (T,), dtype=_torch.long)
        args = (logits, targets, V_local, rank, world_size,
                num_devices, cores_per_device, _StubXM, _torch)
        return args, {"num_nodes": num_nodes}

    return None


def _training_shape_gate(problem, evolved_code, verbose=False):
    """Try-execute the candidate's evolved fn against a training-shape
    fixture. Returns (ok: bool, error_msg: str).

    Any exception (shape mismatch, missing attr, unsupported op, etc.)
    disqualifies the candidate. A `None` fixture (problem has none
    registered) returns ok=True (gate is skipped, preserving the
    existing flow).
    """
    fixture = _build_training_shape_fixture(problem.name)
    if fixture is None:
        return True, ""
    args, kwargs = fixture
    try:
        ns = {}
        # Provide a real torch and a stub xm/torch_xla so any
        # `import torch_xla...` inside the candidate succeeds.
        import types as _types
        import torch as _torch
        ns_globals = {
            "__builtins__": __builtins__,
            "torch": _torch,
        }
        _xla_mod = _types.ModuleType("torch_xla")
        _xla_core = _types.ModuleType("torch_xla.core")
        _xla_xm = _types.ModuleType("torch_xla.core.xla_model")
        for k in dir(_StubXM):
            if not k.startswith("_"):
                setattr(_xla_xm, k, getattr(_StubXM, k))
        _xla_rt = _types.ModuleType("torch_xla.runtime")
        _xla_rt.global_ordinal = lambda: 0
        _xla_rt.world_size = lambda: 224
        _xla_core.xla_model = _xla_xm
        _xla_mod.core = _xla_core
        _xla_mod.runtime = _xla_rt
        import sys as _sys
        _saved = {k: _sys.modules.get(k) for k in (
            "torch_xla", "torch_xla.core",
            "torch_xla.core.xla_model", "torch_xla.runtime")}
        _sys.modules["torch_xla"] = _xla_mod
        _sys.modules["torch_xla.core"] = _xla_core
        _sys.modules["torch_xla.core.xla_model"] = _xla_xm
        _sys.modules["torch_xla.runtime"] = _xla_rt
        try:
            exec(evolved_code, ns_globals, ns)
        finally:
            for k, v in _saved.items():
                if v is None:
                    _sys.modules.pop(k, None)
                else:
                    _sys.modules[k] = v
        fn = ns.get(problem.evolved_fn_name) or \
             ns_globals.get(problem.evolved_fn_name)
        if fn is None:
            return False, (f"evolved fn '{problem.evolved_fn_name}' "
                           f"not defined in candidate code")
        # Single call on training shapes; any raise == disqualified.
        fn(*args, **kwargs)
        return True, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"    [shape-gate FAIL] {msg[:200]}")
        return False, msg[:300]


def phase4b_unconditional_shape_gate(problem, all_results, verbose=True):
    """Run _training_shape_gate on EVERY candidate in all_results,
    regardless of --hw-eval. Tags failing candidates with
    `_hw_gate_failed`/`_hw_gate_error` so Phase 5 winner selection can
    skip them. Idempotent: candidates already gated (e.g. by Phase 4
    when --hw-eval is set) are not re-tested.
    """
    print("\n" + "=" * 70)
    print(f"[Phase 4b] Training-shape gate (all candidates): "
          f"{problem.display_name}")
    print("=" * 70)
    n_total = len(all_results)
    n_pass = n_fail = n_skip = 0
    for name, metrics in all_results:
        if "_hw_gate_failed" in metrics:
            n_skip += 1
            continue
        evolved_code = metrics["_params"].get("evolved_code",
                       metrics["_params"].get("builtin_code", ""))
        if not evolved_code:
            n_skip += 1
            continue
        ok, err = _training_shape_gate(problem, evolved_code,
                                       verbose=False)
        if not ok:
            metrics["_hw_gate_failed"] = True
            metrics["_hw_gate_error"] = err
            n_fail += 1
            if verbose:
                print(f"  FAIL {name}: {err[:140]}")
        else:
            n_pass += 1
            if verbose:
                print(f"  PASS {name}")
    print(f"  Summary: {n_pass} pass / {n_fail} fail / {n_skip} skip "
          f"(total {n_total})")


def phase4_generic_hardware_eval(problem, all_results, topology, num_nodes,
                                  worker_addrs, verbose=True,
                                  use_llm=False, llm_model="sonnet"):
    """Phase 4 for non-alltoallv problems: run top candidates on real hardware."""
    print("\n" + "=" * 70)
    print(f"[Phase 4] Hardware Benchmark: {problem.display_name}")
    print("=" * 70)

    world = topology.num_cores
    _master_addr = "localhost"
    if num_nodes > 1 and worker_addrs:
        import socket
        _master_addr = socket.gethostbyname(socket.gethostname())

    hw_results = []
    seen = set()

    for name, metrics in all_results[:6]:
        tmpl = metrics.get("template", "?")
        if tmpl in seen:
            continue
        seen.add(tmpl)

        evolved_code = metrics["_params"].get("evolved_code",
                       metrics["_params"].get("builtin_code", ""))
        if not evolved_code:
            print(f"  Skip {name}: no code")
            continue

        sim_us = metrics.get("sim_time_us", 0)
        print(f"  Running {name} ({tmpl}, sim={sim_us:.1f} us)...")

        # R20 training-shape pre-gate: call evolved fn on CPU with the
        # actual shapes the deployed runtime will see at training time
        # (heterogeneous OLMoE rep_params, Llama-amp3 micro-batch acts,
        # etc.). Disqualify on any exception BEFORE paying for the
        # multi-node HW microbench. Tags the candidate so Phase 5 can
        # skip it during winner selection.
        ok, gate_err = _training_shape_gate(problem, evolved_code,
                                            verbose=verbose)
        if not ok:
            print(f"    Training-shape gate: FAILED ({gate_err[:160]})")
            metrics["_hw_gate_failed"] = True
            metrics["_hw_gate_error"] = gate_err
            hw_results.append((name, None, sim_us))
            continue

        hw = _run_generic_on_hardware(
            problem, evolved_code, world,
            topology.num_devices, num_nodes,
            _master_addr, worker_addrs)

        if hw and hw.get("hw_latency_ms"):
            hw_ms = hw["hw_latency_ms"]
            print(f"    HW: {hw_ms:.3f} ms | Sim: {sim_us:.1f} us")
            # Training validation gate. We also record the per-step
            # training latency (when the harness reports it) and prefer
            # it for downstream ranking, since the 20-iter isolated-call
            # microbench above does not reflect framework / cache
            # behavior under NEURON_NUM_RECENT_MODELS_TO_KEEP=1.
            if problem.training_validation_code:
                print(f"    Running training validation (10 steps, bf16)...")
                tv = _run_training_validation(
                    problem, evolved_code, world,
                    topology.num_devices, num_nodes,
                    _master_addr, worker_addrs)
                if tv.get("passed"):
                    tv_us = tv.get("avg_step_us")
                    rank_ms = (tv_us / 1000.0) if tv_us else hw_ms
                    if tv_us:
                        print(f"    Training validation: PASSED "
                              f"(per-step avg = {tv_us/1000.0:.3f} ms)")
                    else:
                        print(f"    Training validation: PASSED")
                    hw_results.append((name, rank_ms, sim_us))
                else:
                    err_msg = tv.get('error', 'unknown')[:200]
                    print(f"    Training validation: FAILED ({err_msg})")
                    if use_llm:
                        fixed = _training_failure_recovery(
                            problem, evolved_code, tv,
                            topology, num_nodes, _master_addr,
                            worker_addrs, llm_model, verbose)
                        if fixed:
                            fixed_code, fixed_hw_ms = fixed
                            hw_results.append(
                                (name + ":tv_fix", fixed_hw_ms, sim_us))
                            metrics["_params"]["evolved_code"] = fixed_code
                            print(f"    Recovery: PASSED ({fixed_hw_ms:.3f} ms)")
            else:
                hw_results.append((name, hw_ms, sim_us))
        else:
            err = hw.get("error", "unknown") if hw else "failed"
            print(f"    FAILED: {err}")

    if hw_results:
        print(f"\n  {'Algorithm':<35} {'Rank (ms)':<11} {'Sim (us)':<10}")
        print("  " + "-" * 55)
        for name, hw_ms, sim_us in hw_results:
            hw_s = f"{hw_ms:<11.3f}" if hw_ms is not None else f"{'GATE-FAIL':<11}"
            sim_s = f"{sim_us:<10.1f}" if sim_us is not None else f"{'n/a':<10}"
            print(f"  {name:<35} {hw_s} {sim_s}")

    return hw_results


# ================================================================
# Phase 5: Final Code Generation → trainium_alltoallv.py
# ================================================================

def phase5_codegen(all_results, send_counts, topology, num_nodes,
                   output_dir, hw_results=None, verbose=True):
    """
    Generate final runtime code from the best candidate.

    Uses HW results to pick the winner if available, otherwise uses
    simulator ranking.

    Generates:
    - runtime/trainium_alltoallv.py (importable module)
    - experiments/results/best_<pattern>_<template>.py (standalone benchmark)
    - experiments/results/results_<pattern>.json (full results)
    """
    print("\n" + "=" * 70)
    print("[Phase 5] Final Code Generation → trainium_alltoallv.py")
    print("=" * 70)

    world = topology.num_cores
    num_devices = topology.num_devices

    # Pick winner. The hw_results entries that arrived here have already
    # passed both the on-Trainium HW microbench and the 10-step bf16
    # training validation, i.e., they are correct AND feasible. Among
    # those survivors we rank by SIMULATOR score, not HW microbench
    # latency:
    #
    # - HW microbench measures isolated-call latency (20 iter, fresh
    #   tensors, warm NEFF cache). It does NOT reflect 5000-step training
    #   physics where NEFF cache eviction, framework overhead, and
    #   per-mark_step graph_launch dominate.
    # - The simulator IS a per-step training cost predictor: it accounts
    #   for graph_launch_overhead, NEFF compilation cost amortized over
    #   training steps, contiguity-aware implicit copies, and per-op
    #   work proportional to data volume. When microbench and simulator
    #   disagree, the simulator's prediction tracks 5000-step training
    #   latency better.
    #
    # The hw filter (median/5) is kept as a sanity check that drops
    # candidates whose microbench latency is suspiciously low (likely
    # a benchmark that didn't actually run the full work).
    winner_name, winner_m = all_results[0]
    if hw_results:
        hw_sorted = sorted(
            [(name, hw_ms) for name, hw_ms, _ in hw_results if hw_ms],
            key=lambda x: x[1])
        if len(hw_sorted) >= 2:
            median_hw = sorted(h[1] for h in hw_sorted)[len(hw_sorted) // 2]
            hw_sorted = [(n, h) for n, h in hw_sorted
                         if h > median_hw / 5.0]
        survivor_names = {n for (n, _h) in hw_sorted}
        if survivor_names:
            survivors = [(name, m) for name, m in all_results
                         if name in survivor_names]
            survivors.sort(key=lambda x: x[1].get("sim_time_us", float("inf")))
            if survivors:
                winner_name, winner_m = survivors[0]
                print(f"  Winner selected by simulator score among "
                      f"HW+TV survivors: {winner_name} "
                      f"(sim={winner_m['sim_time_us']:.1f} us)")
        else:
            # No candidate passed Phase-4 HW eval (all tested candidates
            # failed training-validation). The simulator's #1 sim winner
            # is HW-broken; fall back to the best-sim baseline template
            # (a human-written starting point known to compile + run).
            failed_names = {name for name, hw_ms, _ in hw_results
                             if hw_ms is None}
            baseline_candidates = [
                (name, m) for name, m in all_results
                if name.startswith("baseline:") and name not in failed_names
            ]
            baseline_candidates.sort(
                key=lambda x: x[1].get("sim_time_us", float("inf")))
            if baseline_candidates:
                winner_name, winner_m = baseline_candidates[0]
                print(f"  All HW+TV survivors empty; fell back to "
                      f"best baseline template: {winner_name} "
                      f"(sim={winner_m['sim_time_us']:.1f} us). "
                      f"Tested-and-failed: "
                      f"{sorted(failed_names)}")

    winner_template = winner_m["template"]
    winner_params = winner_m["_params"]

    print(f"  Winner: {winner_name} (template={winner_template})")
    print(f"  Score: {winner_m['cost_score']:.3f}, "
          f"SimTime: {winner_m['sim_time_us']:.1f} us")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        code = generate_trainium_code(winner_template, winner_params,
                                       num_nodes=num_nodes)
        (out / f"best_moe_{winner_template}.py").write_text(code)

        serializable = []
        for name, metrics in all_results:
            entry = {k: v for k, v in metrics.items()
                     if k != "_params" and not k.startswith("_")}
            entry["name"] = name
            p = metrics.get("_params", {})
            entry["params"] = {k: v for k, v in p.items()
                               if not k.startswith("_")}
            serializable.append(entry)
        (out / "results_moe.json").write_text(json.dumps(serializable, indent=2))

    # Generate Python wrapper into runtime/
    if output_dir:
        wrapper_code = emit_python_wrapper(
            winner_template, winner_params,
            world=world, num_devices=num_devices, num_nodes=num_nodes)
        runtime_dir = Path("runtime")
        runtime_dir.mkdir(exist_ok=True)
        wrapper_path = runtime_dir / (
            "trainium_alltoallv.py" if num_nodes <= 1
            else f"trainium_alltoallv_{num_nodes}node.py")
        wrapper_path.write_text(wrapper_code)
        print(f"\n  Generated: {wrapper_path}")
        print(f"  To use:   from runtime import all_to_allv, init_alltoallv")

    return winner_name, winner_m


# ================================================================
# Main orchestrator
# ================================================================

def run_search(pattern="moe", use_llm=True, llm_model="opus",
               phase3_style="multi-island",
               llm_candidates=3, ga_generations=200, ga_population=100,
               sa_iters=5000, hw_eval=False, output_dir=None,
               num_nodes=1, worker_addrs=None,
               problem_name="alltoallv", max_rounds=8,
               no_simulator=False):
    """
    Run the 5-phase collective communication search pipeline.

    Phase 1: Agent Hardware Profiling (LLM builds simulator)
    Phase 2: Baseline Evaluation on Simulator → Knowledgebase
    Phase 3: Multi-island Evolution with Simulator Feedback
    Phase 4: Iterative Mini-benchmarking + Refining on Real HW
    Phase 5: Final Code Generation → runtime/trainium_<problem>.py

    All problems go through the same 5 phases. For AllToAllV, Phase 2 also
    includes schedule-based GA/SA search and Phase 3 includes island evolution
    and CGIS refinement (these are AllToAllV-specific and complement template
    evolution). For other problems, Phase 2 evaluates builtin templates and
    Phase 3 runs multi-island template evolution only.

    Feedback loops: Phase 2/3 can trigger Phase 1 simulator refinement.
    """
    problem = get_problem(problem_name)

    print("=" * 70)
    print(f"Collective Search Pipeline: {problem.display_name} (5-Phase)")
    print(f"Pattern: {pattern} | LLM: {use_llm} | HW eval: {hw_eval}")
    print("=" * 70)

    # ---- Phase 1: Agent Hardware Profiling ----
    agent_sim, topology, dispatch_overhead = phase1_profiling(
        use_llm, llm_model, num_nodes)

    world = topology.num_cores
    send_counts = make_send_counts(pattern, world=world)

    # ---- Phase 2: Baseline Evaluation → Knowledgebase (unified) ----
    if True:
        op_costs = _extract_op_costs(agent_sim)
        # Pull training-context physics that the agent surfaced in Phase 1.
        # These are no-ops when measurements aren't present (legacy behavior).
        _agent_meas = _HARDWARE_MEASUREMENTS
        _comp_info = _agent_meas.get("compilation_cost_us", {}) or {}
        _comp_samples = _comp_info.get("samples", [])
        _load_events = _comp_info.get("load_events_per_run", 2)
        _amortize_steps = _comp_info.get("typical_training_steps", 5000)
        _glo = _agent_meas.get("graph_launch_overhead_us", {}) or {}
        _glo_us = _glo.get("per_mark_step_us", 0.0)
        # Per-problem training-scale tensor multiplier: scales the simulator's
        # small correctness-test inputs up to the size class typically seen at
        # full training. Sourced from each problem's training_validation_code.
        _train_scale_mult = _problem_train_scale_multiplier(problem)
        # On-device memcpy bandwidth (bytes/us) used to charge the
        # implicit copy that PyTorch silently inserts when reshape() /
        # contiguous() runs on a non-contiguous source. Two regimes:
        # strided (sub-region gather) and sequential (dense permute).
        # Falls back to defaults from static measurements if the agent
        # never called measure_memory_copy_throughput.
        _memcpy_bps, _memcpy_seq_bps = _extract_memcpy_bw(agent_sim)
        _mc_info = _agent_meas.get("memcpy_throughput", {}) or {}
        if _memcpy_bps == 0.0:
            _memcpy_bps = float(_mc_info.get("strided_gbps", 0.0)) * 1000.0
        if _memcpy_seq_bps == 0.0:
            _memcpy_seq_bps = float(_mc_info.get("sequential_gbps", 0.0)) * 1000.0
        print("  [Phase 1 done] pipeline_amort_alphas = ({:.3f}, {:.3f}, {:.3f})".format(
            getattr(agent_sim.config, "pipeline_amort_alpha1", 1.0),
            getattr(agent_sim.config, "pipeline_amort_alpha2", 1.0),
            getattr(agent_sim.config, "pipeline_amort_alpha3", 1.0)))
        baseline_results = phase2_generic_baseline(
            problem, topology, dispatch_overhead, num_nodes,
            unsupported_primitives=agent_sim.config.unsupported_primitives,
            op_costs=op_costs,
            graph_launch_overhead_us=_glo_us,
            compilation_cost_samples=_comp_samples,
            compilation_load_events_per_run=_load_events,
            compilation_amortize_steps=_amortize_steps,
            pipeline_amort_alpha1=getattr(agent_sim.config, "pipeline_amort_alpha1", 0.30),
            pipeline_amort_alpha2=getattr(agent_sim.config, "pipeline_amort_alpha2", 0.10),
            pipeline_amort_alpha3=getattr(agent_sim.config, "pipeline_amort_alpha3", 0.02),
            training_scale_bytes_multiplier=_train_scale_mult,
            memcpy_bytes_per_us=_memcpy_bps,
            memcpy_seq_bytes_per_us=_memcpy_seq_bps,
            standalone_graph_cost_cfg=agent_sim.knowledgebase.get("standalone_graph_cost_us", {}),
            unsupported_local_ops=[p for p in (agent_sim.config.unsupported_primitives or []) if p in {"cumsum", "cumprod", "sort", "argsort"}])
        cost_model = CostModel(topology, send_counts,
                               dispatch_overhead_us=dispatch_overhead,
                               graph_launch_overhead_us=_glo_us,
                               compilation_cost_samples=_comp_samples,
                               compilation_load_events_per_run=_load_events,
                               compilation_amortize_steps=_amortize_steps)
        knowledgebase = _build_knowledgebase(baseline_results, topology, cost_model)
        all_results = baseline_results
        refinement_needed = None

    # ---- Phase 3: Generic Code-gen Evolution (unified) ----
    if True:
        all_results = phase3_generic_evolution_dispatch(
            problem, topology, send_counts, cost_model,
            all_results, use_llm, llm_model,
            num_nodes, max_rounds,
            phase3_style=phase3_style,
            no_simulator=no_simulator,
            unsupported_primitives=agent_sim.config.unsupported_primitives,
            op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead,
            graph_launch_overhead_us=_glo_us,
            compilation_cost_samples=_comp_samples,
            compilation_load_events_per_run=_load_events,
            compilation_amortize_steps=_amortize_steps,
            pipeline_amort_alpha1=getattr(agent_sim.config, "pipeline_amort_alpha1", 0.30),
            pipeline_amort_alpha2=getattr(agent_sim.config, "pipeline_amort_alpha2", 0.10),
            pipeline_amort_alpha3=getattr(agent_sim.config, "pipeline_amort_alpha3", 0.02),
            training_scale_bytes_multiplier=_train_scale_mult,
            memcpy_bytes_per_us=_memcpy_bps,
            memcpy_seq_bytes_per_us=_memcpy_seq_bps)

    # ---- Final ranking before hardware ----
    all_results.sort(key=lambda x: x[1]["cost_score"])
    print("\n" + "=" * 70)
    print("RANKING (post-evolution, pre-hardware)")
    print("=" * 70)
    for i, (name, m) in enumerate(all_results[:15]):
        print(f"  {i+1}. {name}: {m['sim_time_us']:.1f} us, "
              f"{m.get('local_ops', '?')} ops")

    winner_name, winner_m = all_results[0]
    print(f"\nSimulator winner: {winner_name} "
          f"(sim={winner_m['sim_time_us']:.1f} us)")

    # ---- Phase 4: Hardware evaluation ----
    hw_results = None
    if hw_eval:
        hw_results = phase4_generic_hardware_eval(
            problem, all_results, topology, num_nodes, worker_addrs,
            use_llm=use_llm, llm_model=llm_model)

    # ---- Phase 4b: Unconditional training-shape gate ----
    # Runs on EVERY candidate regardless of --hw-eval so Phase 5 never
    # picks a candidate that crashes at training time on cat-shape /
    # missing-attr / unsupported-op bugs. Idempotent w.r.t. Phase 4.
    phase4b_unconditional_shape_gate(problem, all_results)

    # ---- Phase 5: Code generation ----
    if output_dir:
        _ph5_master_addr = "localhost"
        if num_nodes > 1 and worker_addrs:
            import socket
            _ph5_master_addr = socket.gethostbyname(socket.gethostname())
        phase5_generic_codegen(problem, all_results, topology, num_nodes,
                               output_dir, hw_results=hw_results,
                               no_simulator=no_simulator,
                               llm_model=llm_model,
                               master_addr=_ph5_master_addr,
                               worker_addrs=worker_addrs)

    return all_results


MIN_ISLANDS = 3


def _build_island_list(problem, min_islands=MIN_ISLANDS):
    """Build at least `min_islands` starting templates for evolution.

    If the problem has fewer builtins than min_islands, duplicate the first
    builtin so every run has at least 3 independent evolution islands.
    """
    builtins = list(problem.builtin_templates.keys())
    islands = list(builtins)
    idx = 0
    while len(islands) < min_islands:
        islands.append(builtins[idx % len(builtins)])
        idx += 1
    return islands


# ================================================================
# Phase 2 (Generic): Baseline Evaluation on Simulator
# ================================================================

def phase2_generic_baseline(problem, topology, dispatch_overhead, num_nodes,
                            verbose=True, unsupported_primitives=None,
                            op_costs=None,
                            graph_launch_overhead_us=0.0,
                            compilation_cost_samples=None,
                            compilation_load_events_per_run=2,
                            compilation_amortize_steps=5000,
                            pipeline_amort_alpha1=0.30,
                            pipeline_amort_alpha2=0.10,
                            pipeline_amort_alpha3=0.02,
                            training_scale_bytes_multiplier=1.0,
                            memcpy_bytes_per_us=0.0,
                            memcpy_seq_bytes_per_us=0.0,
                            standalone_graph_cost_cfg=None,
                            unsupported_local_ops=None):
    """
    Evaluate all builtin templates on the simulator to establish baselines.

    Returns:
        baseline_results: list of (name, metrics) ranked by score
    """
    from search.correctness_test import benchmark_xla_candidate_generic

    print("\n" + "=" * 70)
    print(f"[Phase 2] Baseline Evaluation: {problem.display_name}")
    print("=" * 70)

    world = topology.num_cores
    send_counts = make_send_counts("moe", world=world)

    baseline_results = []
    for tname, code in problem.builtin_templates.items():
        ns = {}
        exec(code, ns)
        fn = ns[problem.evolved_fn_name]
        # Attach the source for AST-based structural analysis in the cost
        # model (per-mark_step graph tax, bucket-bytes cap detection).
        try:
            fn.__candidate_source__ = code
        except Exception:
            pass
        bench = benchmark_xla_candidate_generic(
            problem, fn, topology, send_counts, world, num_nodes=num_nodes,
            unsupported_primitives=unsupported_primitives,
            op_costs=op_costs, dispatch_overhead_us=dispatch_overhead,
            graph_launch_overhead_us=graph_launch_overhead_us,
            compilation_cost_samples=compilation_cost_samples,
            compilation_load_events_per_run=compilation_load_events_per_run,
            compilation_amortize_steps=compilation_amortize_steps,
            pipeline_amort_alpha1=pipeline_amort_alpha1,
            pipeline_amort_alpha2=pipeline_amort_alpha2,
            pipeline_amort_alpha3=pipeline_amort_alpha3,
            training_scale_bytes_multiplier=training_scale_bytes_multiplier,
            memcpy_bytes_per_us=memcpy_bytes_per_us,
            memcpy_seq_bytes_per_us=memcpy_seq_bytes_per_us,
            standalone_graph_cost_cfg=standalone_graph_cost_cfg,
            unsupported_local_ops=unsupported_local_ops)
        if "error" not in bench:
            sim_us = bench["sim_time_us"]
            baseline_results.append((f"baseline:{tname}", {
                "template": tname,
                "cost_score": sim_us / 100.0,
                "sim_time_us": sim_us,
                "local_ops": bench.get("local_ops", "?"),
                "num_collective_permute": bench.get("num_collective_permute", 0),
                "num_all_gather": bench.get("num_all_gather", 0),
                "num_all_reduce": bench.get("num_all_reduce", 0),
                "_params": {"builtin_code": code},
            }))
            if verbose:
                print(f"  {tname}: {sim_us:.1f} us, "
                      f"{bench.get('local_ops', '?')} local ops, "
                      f"{bench.get('num_all_gather', 0)} ag, "
                      f"{bench.get('num_all_reduce', 0)} ar, "
                      f"{bench.get('num_collective_permute', 0)} cp")
        else:
            print(f"  {tname}: ERROR: {bench['error']}")

    baseline_results.sort(key=lambda x: x[1]["cost_score"])
    if baseline_results:
        print(f"\n  Best baseline: {baseline_results[0][0]} "
              f"({baseline_results[0][1]['sim_time_us']:.1f} us)")

    return baseline_results


# ================================================================
# Phase 3 (Generic): Multi-island Template Evolution
# ================================================================

def phase3_generic_evolution_dispatch(problem, topology, send_counts, cost_model,
                                      baseline_results, use_llm, llm_model,
                                      num_nodes, max_rounds, verbose=True,
                                      phase3_style="multi-island",
                                      **kwargs):
    """Dispatch to multi-island GA (default) or CC-style single-trajectory ReAct."""
    if phase3_style == "strategy-enumerate":
        from search.strategy_enumerate_phase3 import _phase3_strategy_enumerate
        return _phase3_strategy_enumerate(
            problem, topology, send_counts, cost_model, baseline_results,
            use_llm, llm_model, num_nodes, max_rounds,
            verbose=verbose,
            no_simulator=kwargs.pop("no_simulator", False),
            **kwargs)
    # drain no_simulator before passing to non-strat-enum styles
    kwargs.pop("no_simulator", None)
    if phase3_style == "cc-react":
        return _phase3_cc_react(
            problem, topology, send_counts, cost_model, baseline_results,
            use_llm, llm_model, num_nodes, max_rounds,
            verbose=verbose, **kwargs)
    return phase3_generic_evolution(
        problem, topology, send_counts, cost_model, baseline_results,
        use_llm, llm_model, num_nodes, max_rounds,
        verbose=verbose, **kwargs)


def _phase3_cc_react(problem, topology, send_counts, cost_model,
                     baseline_results, use_llm, llm_model,
                     num_nodes, max_rounds,
                     verbose=True, **kwargs):
    """CC-style single-trajectory ReAct phase 3.

    Same LLM, same prompt template, same simulator scoring, same
    correctness gate as multi-island. Only difference: K islands x
    max_rounds is collapsed to a single trajectory of (K * max_rounds)
    turns where the LLM sees the FULL history of prior candidates.
    """
    print("\n" + "=" * 70)
    print(f"[Phase 3 / CC-ReAct] Single-trajectory: {problem.display_name}")
    print("=" * 70)
    all_results = list(baseline_results)
    if not use_llm:
        print("  Skipping evolution (--no-llm)")
        return all_results
    islands_for_budget = _build_island_list(problem)
    total_turns = len(islands_for_budget) * max_rounds
    print(f"  CC-ReAct total turn budget: {total_turns} "
          f"(== multi-island {len(islands_for_budget)} islands x {max_rounds} rounds)")
    starting = islands_for_budget[0]
    print(f"  Starting from baseline: {starting}")
    analyzer = ContentionAnalyzer(topology, send_counts)
    te = TemplateEvolution(
        topology, send_counts, cost_model, analyzer,
        model=llm_model, problem=problem, **kwargs)
    try:
        evo_code, evo_bench, evo_hist = te.evolve(
            starting_template=starting, max_rounds=total_turns,
            verbose=verbose)
        if evo_bench and "sim_time_us" in evo_bench:
            sim_us = evo_bench["sim_time_us"]
            all_results.append((f"cc:final", {
                "template": f"cc_react_final",
                "cost_score": sim_us / 100.0,
                "sim_time_us": sim_us,
                "local_ops": evo_bench.get("local_ops", "?"),
                "num_collective_permute": evo_bench.get("num_collective_permute", 0),
                "num_all_gather": evo_bench.get("num_all_gather", 0),
                "num_all_reduce": evo_bench.get("num_all_reduce", 0),
                "_params": {"evolved_code": evo_code},
            }))
    except Exception as e:
        print(f"  CC-ReAct failed: {e}")
    all_results.sort(key=lambda x: x[1]["cost_score"])
    print(f"\n  CC-ReAct complete. {len(all_results)} total candidates.")
    return all_results


def phase3_generic_evolution(problem, topology, send_counts, cost_model,
                             baseline_results, use_llm, llm_model,
                             num_nodes, max_rounds, verbose=True,
                             unsupported_primitives=None,
                             op_costs=None, dispatch_overhead_us=100.0,
                             graph_launch_overhead_us=0.0,
                             compilation_cost_samples=None,
                             compilation_load_events_per_run=2,
                             compilation_amortize_steps=5000,
                             pipeline_amort_alpha1=0.30,
                             pipeline_amort_alpha2=0.10,
                             pipeline_amort_alpha3=0.02,
                             training_scale_bytes_multiplier=1.0,
                             memcpy_bytes_per_us=0.0,
                             memcpy_seq_bytes_per_us=0.0):
    """
    Multi-island LLM template evolution for any collective problem.

    Runs at least MIN_ISLANDS independent evolution islands, each starting
    from a different builtin template (duplicating if fewer builtins exist).

    Returns:
        all_results: baseline_results + evolved results, ranked
    """
    print("\n" + "=" * 70)
    print(f"[Phase 3] Multi-island Template Evolution: {problem.display_name}")
    print("=" * 70)

    all_results = list(baseline_results)

    if not use_llm:
        print("  Skipping evolution (--no-llm)")
        return all_results

    analyzer = ContentionAnalyzer(topology, send_counts)
    islands = _build_island_list(problem)

    print(f"  Islands ({len(islands)}): {islands}")

    for i, starting in enumerate(islands):
        print(f"\n  --- Island {i+1}/{len(islands)}: {starting} ---")
        te = TemplateEvolution(
            topology, send_counts, cost_model, analyzer,
            model=llm_model, problem=problem,
            unsupported_primitives=unsupported_primitives,
            op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead_us,
            graph_launch_overhead_us=graph_launch_overhead_us,
            compilation_cost_samples=compilation_cost_samples,
            compilation_load_events_per_run=compilation_load_events_per_run,
            compilation_amortize_steps=compilation_amortize_steps,
            pipeline_amort_alpha1=pipeline_amort_alpha1,
            pipeline_amort_alpha2=pipeline_amort_alpha2,
            pipeline_amort_alpha3=pipeline_amort_alpha3,
            training_scale_bytes_multiplier=training_scale_bytes_multiplier,
            memcpy_bytes_per_us=memcpy_bytes_per_us,
            memcpy_seq_bytes_per_us=memcpy_seq_bytes_per_us)
        try:
            evo_code, evo_bench, evo_hist = te.evolve(
                starting_template=starting, max_rounds=max_rounds,
                verbose=verbose)
            if evo_bench and "sim_time_us" in evo_bench:
                sim_us = evo_bench["sim_time_us"]
                all_results.append((f"evo:{starting}_{i}", {
                    "template": f"evolved_{starting}",
                    "cost_score": sim_us / 100.0,
                    "sim_time_us": sim_us,
                    "local_ops": evo_bench.get("local_ops", "?"),
                    "num_collective_permute": evo_bench.get("num_collective_permute", 0),
                    "num_all_gather": evo_bench.get("num_all_gather", 0),
                    "num_all_reduce": evo_bench.get("num_all_reduce", 0),
                    "_params": {"evolved_code": evo_code},
                }))
        except Exception as e:
            print(f"  Island {i+1} ({starting}) failed: {e}")

    all_results.sort(key=lambda x: x[1]["cost_score"])

    print(f"\n  Evolution complete. {len(all_results)} total candidates.")
    if all_results:
        w = all_results[0]
        print(f"  Best: {w[0]} ({w[1]['sim_time_us']:.1f} us, "
              f"{w[1].get('local_ops', '?')} local ops)")

    return all_results



# ================================================================
# Phase 5 (Generic): Code Generation → runtime/trainium_<problem>.py
# ================================================================

def _llm_judge_pick_by_ss(problem, survivor_results, ss_results, llm_model="sonnet"):
    """Ask the LLM to pick a winner among Phase-4 HW survivors using
    small-shape HW microbench latencies + code snippets. Returns the
    candidate name (or None on parse failure). No simulator info is
    shown to the LLM."""
    from search.template_evolution import _invoke_bedrock
    lines = []
    for name, m in survivor_results:
        code = m["_params"].get("evolved_code") or m["_params"].get("builtin_code", "")
        ss_ms = ss_results.get(name)
        if ss_ms is None:
            continue
        lines.append(
            f"- {name}: small-shape HW microbench = {ss_ms:.3f} ms\n"
            f"  Code (first 800 chars):\n{code[:800]}\n")
    if not lines:
        return None
    prompt = (
        f"You are picking the best candidate for {problem.display_name} "
        f"on a 7-node trn1 cluster. The simulator was disabled for this "
        f"ablation; you must decide using small-shape HW microbench "
        f"latencies (h7-style probe, mc=20) plus visible code structure.\n\n"
        f"Candidates that passed Phase-4 hardware shape-gate:\n\n"
        + "\n".join(lines)
        + "\n\nPick the candidate name (one of the bullets above) that "
        f"you judge to be the best to deploy. Prefer simpler patterns "
        f"with fewer Python-level loops and fewer collective dispatches "
        f"when latencies are within noise. Reply with ONLY the exact "
        f"candidate name string from the bullets, nothing else."
    )
    try:
        resp = _invoke_bedrock(prompt, model=llm_model, temperature=0.0, max_tokens=200)
    except Exception as e:
        print(f"  [LLM-judge] invoke error: {e}")
        return None
    # Look for an exact-prefix match against known candidate names.
    resp = resp.strip().splitlines()[0].strip() if resp.strip() else ""
    names = [n for n, _ in survivor_results]
    for n in names:
        if resp.startswith(n) or n in resp:
            return n
    return None


def phase5_generic_codegen(problem, all_results, topology, num_nodes,
                           output_dir, hw_results=None, verbose=True,
                           no_simulator=False, llm_model="sonnet",
                           master_addr=None, worker_addrs=None):
    """
    Generate final runtime code from the best candidate for any problem.
    """
    print("\n" + "=" * 70)
    print(f"[Phase 5] Code Generation → runtime/{problem.runtime_module_name}.py")
    print("=" * 70)

    world = topology.num_cores

    # Among hw_results survivors (passed HW microbench AND 10-step
    # training validation), rank by SIMULATOR score. The HW microbench
    # measures isolated-call latency and does not reflect 5000-step
    # training physics (NEFF cache eviction under
    # NEURON_NUM_RECENT_MODELS_TO_KEEP=1, framework overhead per
    # mark_step). The simulator's per-step prediction is closer to
    # actual training latency. The hw filter (median/5) drops candidates
    # whose microbench is suspiciously low (likely a benchmark that did
    # not actually run the full work).
    # R20: never pick a winner that we know failed the training-shape
    # HW gate. Build the set of disqualified names up front; restrict
    # the simulator-ranked candidate list to those that did NOT fail
    # the gate (whether they were even tested or not).
    # R20-fixup: previously failed_names was restricted to candidates
    # in hw_results, missing Phase-4b gate failures when --hw-eval is
    # off. Read _hw_gate_failed directly from every candidate.
    failed_names = {n for n, m in all_results if m.get("_hw_gate_failed")}
    gate_passed_results = [(n, m) for n, m in all_results
                           if n not in failed_names]
    if not gate_passed_results:
        gate_passed_results = all_results  # safety: fall back to all

    winner_name, winner_m = gate_passed_results[0]
    if hw_results:
        hw_sorted = sorted(
            [(name, hw_ms) for name, hw_ms, _ in hw_results if hw_ms],
            key=lambda x: x[1])
        if len(hw_sorted) >= 2:
            median_hw = sorted(h[1] for h in hw_sorted)[len(hw_sorted) // 2]
            hw_sorted = [(n, h) for n, h in hw_sorted
                         if h > median_hw / 5.0]
        survivor_names = {n for (n, _h) in hw_sorted}
        if survivor_names:
            survivors = [(name, m) for name, m in gate_passed_results
                         if name in survivor_names]
            if no_simulator and master_addr is not None:
                print(f"\n  [NO-SIM] Phase 5 small-shape microbench on "
                      f"{len(survivors)} survivors...")
                ss_results = {}
                for s_name, s_m in survivors:
                    s_code = s_m["_params"].get("evolved_code") or \
                             s_m["_params"].get("builtin_code", "")
                    if not s_code:
                        continue
                    ss = _run_small_shape_microbench(
                        problem, s_code, topology.num_cores,
                        topology.num_devices, num_nodes,
                        master_addr, worker_addrs)
                    if ss and ss.get("hw_latency_ms") is not None:
                        ss_results[s_name] = ss["hw_latency_ms"]
                        print(f"    {s_name}: small-shape = {ss['hw_latency_ms']:.3f} ms")
                    else:
                        print(f"    {s_name}: small-shape FAILED ({ss.get('error', 'unknown')[:120]})")
                # Rank by small-shape latency
                ss_sorted = sorted(ss_results.items(), key=lambda x: x[1])
                if ss_sorted:
                    print(f"  [NO-SIM] Small-shape ranking:")
                    for n, ms in ss_sorted:
                        print(f"    {n}: {ms:.3f} ms")
                # LLM-judge picks among survivors with small-shape numbers
                llm_pick = _llm_judge_pick_by_ss(
                    problem, [(n, m) for n, m in survivors if n in ss_results],
                    ss_results, llm_model=llm_model)
                if llm_pick:
                    print(f"  [NO-SIM] LLM-judge picked: {llm_pick}")
                    chosen = next(((n, m) for n, m in survivors if n == llm_pick), None)
                    if chosen:
                        winner_name, winner_m = chosen
                    elif ss_sorted:
                        winner_name = ss_sorted[0][0]
                        winner_m = dict(survivors)[winner_name]
                        print(f"  [NO-SIM] Pick not in survivors; using small-shape best: {winner_name}")
                elif ss_sorted:
                    winner_name = ss_sorted[0][0]
                    winner_m = dict(survivors)[winner_name]
                    print(f"  [NO-SIM] LLM-judge failed; using small-shape best: {winner_name}")
                else:
                    # No small-shape data; fall back to simulator-best
                    survivors.sort(key=lambda x: x[1].get("sim_time_us", float("inf")))
                    winner_name, winner_m = survivors[0]
                    print(f"  [NO-SIM] No small-shape data; fallback simulator-best: {winner_name}")
            else:
                survivors.sort(key=lambda x: x[1].get("sim_time_us", float("inf")))
                if survivors:
                    winner_name, winner_m = survivors[0]
                    print(f"  Winner selected by simulator score among "
                          f"HW+TV survivors: {winner_name} "
                          f"(sim={winner_m['sim_time_us']:.1f} us)")
        elif failed_names:
            # All Phase-4-tested candidates failed the shape gate; the
            # sim winner is HW-broken. Fall back to the best-sim
            # candidate that wasn't disqualified.
            print(f"  All Phase-4 candidates failed HW shape-gate: "
                  f"{sorted(failed_names)}. Falling back to next-best "
                  f"simulator candidate.")
            winner_name, winner_m = gate_passed_results[0]
            print(f"  Fallback winner: {winner_name} "
                  f"(sim={winner_m['sim_time_us']:.1f} us)")
    print(f"  Winner: {winner_name}")
    print(f"  Score: {winner_m['cost_score']:.3f}, "
          f"SimTime: {winner_m['sim_time_us']:.1f} us, "
          f"Local ops: {winner_m.get('local_ops', '?')}")

    evolved_code = winner_m["_params"].get("evolved_code",
                   winner_m["_params"].get("builtin_code", ""))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        serializable = []
        for name, metrics in all_results:
            entry = {k: v for k, v in metrics.items()
                     if k != "_params" and not k.startswith("_")}
            entry["name"] = name
            serializable.append(entry)
        results_file = out / f"results_{problem.name}.json"
        results_file.write_text(json.dumps(serializable, indent=2))
        print(f"  Results: {results_file}")

    runtime_dir = Path("runtime")
    runtime_dir.mkdir(exist_ok=True)
    wrapper = _emit_collective_runtime(
        problem, evolved_code, world,
        topology.num_devices, num_nodes)
    runtime_path = runtime_dir / (
        f"{problem.runtime_module_name}.py" if num_nodes <= 1
        else f"{problem.runtime_module_name}_{num_nodes}node.py")
    runtime_path.write_text(wrapper)
    print(f"  Generated: {runtime_path}")

    return winner_name, winner_m


def _emit_collective_runtime(problem, evolved_code, world, num_devices,
                              num_nodes):
    """Generate a runtime module for any collective problem."""
    cpd = 2
    api_section = ""
    if problem.public_api_code:
        api_section = f"""

# ================================================================
# Public API
# ================================================================

{problem.public_api_code}"""

    return f'''"""
{problem.display_name}: Optimized implementation for AWS Trainium.

Generated by the LLM-guided algorithm evolution agent.
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_WORLD = {world}
_NUM_DEVICES = {num_devices}
_NUM_NODES = {num_nodes}
_CORES_PER_DEVICE = {cpd}

_rank = None
_world_size = None


def init_{problem.name}():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


# Agent-evolved algorithm
{evolved_code}
{api_section}
'''


def main():
    parser = argparse.ArgumentParser(
        description="Collective Communication Search for Trainium")
    parser.add_argument("--problem", default="alltoallv",
                        choices=list(PROBLEMS.keys()),
                        help="Collective problem to optimize")
    parser.add_argument("--pattern", default="moe",
                        choices=["moe", "uniform", "skewed", "sparse", "random",
                                 "increasing", "locality"])
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", default="opus", choices=["haiku", "sonnet", "opus"])
    parser.add_argument("--llm-candidates", type=int, default=3)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--sa-iters", type=int, default=5000)
    parser.add_argument("--hw-eval", action="store_true")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--all-patterns", action="store_true")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--worker-addrs", default="",
                        help="Comma-separated private IPs of worker nodes")
    parser.add_argument("--max-rounds", type=int, default=8,
                        help="Max evolution rounds per island")
    parser.add_argument("--phase3-style", default="multi-island",
                        choices=["multi-island", "cc-react", "strategy-enumerate"],
                        help="Phase 3 search shape")
    parser.add_argument("--no-simulator", action="store_true",
                        help="Ablation: skip simulator-guided refinement; "
                             "pick the first correct enumerated strategy.")
    args = parser.parse_args()

    patterns = (["moe", "skewed", "sparse", "random", "increasing", "locality"]
                if args.all_patterns else [args.pattern])

    _worker_addrs = ([a.strip() for a in args.worker_addrs.split(",")
                       if a.strip()] if args.worker_addrs else None)

    for pattern in patterns:
        run_search(
            pattern=pattern,
            use_llm=not args.no_llm,
            phase3_style=args.phase3_style,
            llm_model=args.llm_model,
            llm_candidates=args.llm_candidates,
            ga_generations=args.generations,
            ga_population=args.population,
            sa_iters=args.sa_iters,
            hw_eval=args.hw_eval,
            output_dir=args.output_dir,
            num_nodes=args.num_nodes,
            worker_addrs=_worker_addrs,
            problem_name=args.problem,
            max_rounds=args.max_rounds,
            no_simulator=args.no_simulator,
        )
        print()


if __name__ == "__main__":
    main()
