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

    if use_llm:
        print(f"  Model: {llm_model} | max_turns: 15")
        try:
            agent_sim = run_profiling_agent(model=llm_model, verbose=verbose)
            config = agent_sim.config
            if config.is_complete():
                dispatch_overhead = config.collective_dispatch_overhead_us
                print(f"\n  Agent discovered:")
                print(f"    dispatch_overhead = {dispatch_overhead:.1f} us")
                print(f"    devices = {config.num_devices}, cores/dev = {config.cores_per_device}")
                print(f"    simulator built = {agent_sim.cost_function is not None}")
                print(f"    simulator validated = {agent_sim.is_validated()}")
            else:
                print(f"  Agent config incomplete, using defaults")
        except Exception as e:
            print(f"  Agent profiling failed ({e}), using defaults")
    else:
        print("  Skipping LLM profiling (--no-llm), using defaults")

    # Use agent-discovered topology structure (num_devices, adjacency) but
    # apply sanity checks on bandwidth/latency — the agent often mis-derives
    # these from small-message measurements where fixed overhead dominates.
    config = agent_sim.config

    # Discover unsupported primitives via compilation test if not yet populated
    if not config.unsupported_primitives:
        from search.agent_simulator_config import _test_primitive_compilation
        for prim in ["all_gather", "reduce_scatter", "all_reduce",
                     "collective_permute", "all_to_all"]:
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

    # --- GA refinement ---
    print(f"\n  [2b] GA refinement ({ga_generations} gens, pop={ga_population})...")

    def pr_cost(sched):
        s, _ = cost_model.evaluate_template("permute_ring", {"schedule": sched})
        return s

    pr_seeds = [r[1]["_params"]["schedule"] for r in all_results
                if r[1].get("template") == "permute_ring"][:5]
    ga_pr, ga_pr_cost, _ = genetic_search(
        pr_cost, world, ga_population, ga_generations, seed_schedules=pr_seeds)
    m = evaluate_template("permute_ring", {"schedule": ga_pr}, send_counts, topology)
    m["_params"] = {"schedule": ga_pr}
    all_results.append(("pr:GA_refined", m))

    def hier_cost(sched):
        s, _ = cost_model.evaluate_template("hierarchical", {"inter_schedule": sched})
        return s

    hier_seeds = [r[1]["_params"]["inter_schedule"] for r in all_results
                  if r[1].get("template") == "hierarchical"
                  and "inter_schedule" in r[1].get("_params", {})][:3]
    ga_hier, ga_hier_cost, _ = genetic_search(
        hier_cost, num_devices, ga_population, ga_generations,
        seed_schedules=hier_seeds)
    m = evaluate_template("hierarchical", {"inter_schedule": ga_hier},
                          send_counts, topology)
    m["_params"] = {"inter_schedule": ga_hier}
    all_results.append(("hier:GA_refined", m))

    matchings = _generate_matchings(world)
    pw_elements = list(range(world - 1))

    def pw_cost(order):
        s, _ = cost_model.evaluate_template(
            "pairwise", {"round_order": order, "_matchings": matchings})
        return s

    ga_pw, ga_pw_cost, _ = genetic_search(
        pw_cost, world, ga_population, ga_generations, elements=pw_elements)
    pw_params_ga = {"round_order": ga_pw, "_matchings": matchings}
    m = evaluate_template("pairwise", pw_params_ga, send_counts, topology)
    m["_params"] = pw_params_ga
    all_results.append(("pw:GA_refined", m))

    hyb_base = hybrid_default_params(topology, world)
    if hyb_base["far_distances"]:
        def hyb_cost(sched):
            p = dict(hyb_base)
            p["permute_schedule"] = sched
            s, _ = cost_model.evaluate_template("hybrid_ag_perm", p)
            return s

        ga_hyb, ga_hyb_cost, _ = genetic_search(
            hyb_cost, world=world,
            population_size=ga_population, generations=ga_generations,
            seed_schedules=[hyb_base["far_distances"]],
            elements=hyb_base["far_distances"])
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
        for starting in ["naive_allgather", "allgather_reduce_scatter",
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
                              num_nodes, master_addr, worker_addrs, timeout=120):
    """Run a generic collective candidate on real hardware."""
    import os
    import subprocess
    import tempfile
    import re as _re

    NEURON_VENV = os.environ.get(
        "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
    MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

    code = f'''\
#!/usr/bin/env python3
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm
import torch_xla.runtime as xr, torch.distributed as dist

{evolved_code.strip()}

def main():
    device = xla.device()
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    world = xr.world_size()
    rank = xr.global_ordinal()
    num_devices = world // 2
    cpd = 2
    num_nodes = {num_nodes}
'''

    if problem.name == "uniform_a2a":
        code += f'''
    chunk_size = 1024
    x = torch.randn(world * chunk_size, device=device, dtype=torch.float32, requires_grad=True)
    out = {problem.evolved_fn_name}(x, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * chunk_size, device=device, dtype=torch.float32, requires_grad=True)
        out = {problem.evolved_fn_name}(x_i, chunk_size, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
    xm.wait_device_ops()
    end = time.time()
    if rank == 0:
        print(f"latency: {{(end-start)/iters*1000:.3f}} ms")

if __name__ == "__main__":
    main()
'''
    elif problem.name == "fused_reducescatter":
        code += f'''
    tensors = [torch.randn(1024, device=device, dtype=torch.float32, requires_grad=True) for _ in range(8)]
    out = {problem.evolved_fn_name}(tensors, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        ts = [torch.randn(1024, device=device, dtype=torch.float32, requires_grad=True) for _ in range(8)]
        out = {problem.evolved_fn_name}(ts, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
    xm.wait_device_ops()
    end = time.time()
    if rank == 0:
        print(f"latency: {{(end-start)/iters*1000:.3f}} ms")

if __name__ == "__main__":
    main()
'''
    elif problem.name == "ring_kv":
        code += f'''
    kv_chunk = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
    out = {problem.evolved_fn_name}(kv_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        kv_i = torch.randn(2048, device=device, dtype=torch.float32, requires_grad=True)
        out = {problem.evolved_fn_name}(kv_i, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
    xm.wait_device_ops()
    end = time.time()
    if rank == 0:
        print(f"latency: {{(end-start)/iters*1000:.3f}} ms")

if __name__ == "__main__":
    main()
'''
    elif problem.name == "alltoallv":
        code += f'''
    max_chunk = 1024
    send_counts = [max_chunk] * world
    recv_counts = [max_chunk] * world
    x = torch.randn(world * max_chunk, device=device, dtype=torch.float32, requires_grad=True)
    out = {problem.evolved_fn_name}(x, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
    out.sum().backward()
    xla.step()
    iters = 20
    xm.rendezvous("pre_bench")
    start = time.time()
    for _ in range(iters):
        x_i = torch.randn(world * max_chunk, device=device, dtype=torch.float32, requires_grad=True)
        out = {problem.evolved_fn_name}(x_i, send_counts, recv_counts, max_chunk, rank, world, num_devices, cpd, xm, torch, num_nodes=num_nodes)
        out.sum().backward()
        xla.step()
    xm.wait_device_ops()
    end = time.time()
    if rank == 0:
        print(f"latency: {{(end-start)/iters*1000:.3f}} ms")

if __name__ == "__main__":
    main()
'''
    else:
        return {{"error": f"No HW benchmark template for {problem.name}"}}

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

        if num_nodes > 1 and worker_addrs:
            from search.evaluate_algo import _run_multinode_hw
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu")
            output = result.stdout + result.stderr

        match = _re.search(r"latency:\s*([\d.]+)\s*ms", output)
        if match:
            return {
                "hw_latency_ms": float(match.group(1)),
                "output": output[-500:],
            }
        return {"hw_latency_ms": None, "error": "Could not parse latency",
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
    parts.append("    device = xla.device()")
    parts.append("    if not dist.is_initialized():")
    parts.append("        dist.init_process_group('xla', init_method='xla://')")
    parts.append("    rank = xr.global_ordinal()")
    parts.append(f"    init_{problem.name}()")
    parts.append("")
    parts.append("    DIM = 1024")
    parts.append("    model = _Model(DIM, N_LAYERS).to(torch.bfloat16).to(device)")
    parts.append("    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)")
    parts.append("")
    parts.append("    for step in range(N_STEPS):")
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
    parts.append("    if rank == 0:")
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

        if num_nodes > 1 and worker_addrs:
            from search.evaluate_algo import _run_multinode_hw
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu")
            output = result.stdout + result.stderr

        if "TRAINING_VALIDATION_PASSED" in output:
            return {"passed": True, "output": output[-500:]}
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

        hw = _run_generic_on_hardware(
            problem, evolved_code, world,
            topology.num_devices, num_nodes,
            _master_addr, worker_addrs)

        if hw and hw.get("hw_latency_ms"):
            hw_ms = hw["hw_latency_ms"]
            print(f"    HW: {hw_ms:.3f} ms | Sim: {sim_us:.1f} us")
            # Training validation gate
            if problem.training_validation_code:
                print(f"    Running training validation (10 steps, bf16)...")
                tv = _run_training_validation(
                    problem, evolved_code, world,
                    topology.num_devices, num_nodes,
                    _master_addr, worker_addrs)
                if tv.get("passed"):
                    hw_results.append((name, hw_ms, sim_us))
                    print(f"    Training validation: PASSED")
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
        print(f"\n  {'Algorithm':<35} {'HW (ms)':<10} {'Sim (us)':<10}")
        print("  " + "-" * 55)
        for name, hw_ms, sim_us in hw_results:
            print(f"  {name:<35} {hw_ms:<10.3f} {sim_us:<10.1f}")

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

    # Pick winner: prefer HW-validated results (filter outliers)
    winner_name, winner_m = all_results[0]
    if hw_results:
        hw_sorted = sorted(
            [(name, hw_ms) for name, hw_ms, _ in hw_results if hw_ms],
            key=lambda x: x[1])
        if len(hw_sorted) >= 2:
            median_hw = sorted(h[1] for h in hw_sorted)[len(hw_sorted) // 2]
            hw_sorted = [(n, h) for n, h in hw_sorted
                         if h > median_hw / 5.0]
        if hw_sorted:
            hw_winner_name = hw_sorted[0][0]
            for name, m in all_results:
                if name == hw_winner_name:
                    winner_name, winner_m = name, m
                    print(f"  Winner selected by HW benchmark: {winner_name}")
                    break

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
        wrapper_path = runtime_dir / "trainium_alltoallv.py"
        wrapper_path.write_text(wrapper_code)
        print(f"\n  Generated: {wrapper_path}")
        print(f"  To use:   from runtime import all_to_allv, init_alltoallv")

    return winner_name, winner_m


# ================================================================
# Main orchestrator
# ================================================================

def run_search(pattern="moe", use_llm=True, llm_model="opus",
               llm_candidates=3, ga_generations=200, ga_population=100,
               sa_iters=5000, hw_eval=False, output_dir=None,
               num_nodes=1, worker_addrs=None,
               problem_name="alltoallv", max_rounds=8,
               kiss_sorcar=False):
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
    is_alltoallv = (problem_name == "alltoallv")

    print("=" * 70)
    print(f"Collective Search Pipeline: {problem.display_name} (5-Phase)")
    print(f"Pattern: {pattern} | LLM: {use_llm} | HW eval: {hw_eval}")
    print("=" * 70)

    # ---- Phase 1: Agent Hardware Profiling ----
    agent_sim, topology, dispatch_overhead = phase1_profiling(
        use_llm, llm_model, num_nodes)

    world = topology.num_cores
    send_counts = make_send_counts(pattern, world=world)

    # ---- Phase 2: Baseline Evaluation → Knowledgebase ----
    if is_alltoallv:
        all_results, knowledgebase, cost_model, refinement_needed = \
            phase2_baseline_eval(
                topology, send_counts, dispatch_overhead,
                agent_sim, use_llm, llm_model, num_nodes,
                ga_generations, ga_population, sa_iters)

        if refinement_needed and use_llm:
            print(f"\n  [Feedback] Phase 2 → Phase 1: {refinement_needed}")
            agent_sim = refine_simulator(
                agent_sim, refinement_needed, model=llm_model)
            if agent_sim.config.collective_dispatch_overhead_us != dispatch_overhead:
                dispatch_overhead = agent_sim.config.collective_dispatch_overhead_us
                print(f"  Re-running Phase 2 with refined dispatch_overhead={dispatch_overhead:.1f}")
                all_results, knowledgebase, cost_model, _ = \
                    phase2_baseline_eval(
                        topology, send_counts, dispatch_overhead,
                        agent_sim, use_llm, llm_model, num_nodes,
                        ga_generations, ga_population, sa_iters)
    else:
        op_costs = _extract_op_costs(agent_sim)
        baseline_results = phase2_generic_baseline(
            problem, topology, dispatch_overhead, num_nodes,
            unsupported_primitives=agent_sim.config.unsupported_primitives,
            op_costs=op_costs)
        cost_model = CostModel(topology, send_counts,
                               dispatch_overhead_us=dispatch_overhead)
        knowledgebase = _build_knowledgebase(baseline_results, topology, cost_model)
        all_results = baseline_results
        refinement_needed = None

    # ---- Phase 3: Multi-island Evolution ----
    if kiss_sorcar:
        all_results = phase3_kiss_sorcar(
            problem, topology, send_counts, cost_model,
            all_results, num_nodes,
            unsupported_primitives=agent_sim.config.unsupported_primitives)
    elif is_alltoallv:
        all_results, refinement_needed = phase3_evolution(
            topology, send_counts, cost_model, all_results,
            knowledgebase, agent_sim, use_llm, llm_model,
            num_nodes, ga_generations, ga_population,
            dispatch_overhead_us=dispatch_overhead)

        if refinement_needed and use_llm:
            print(f"\n  [Feedback] Phase 3 → Phase 1: {refinement_needed}")
            agent_sim = refine_simulator(
                agent_sim, refinement_needed, model=llm_model)
    else:
        op_costs = _extract_op_costs(agent_sim)
        all_results = phase3_generic_evolution(
            problem, topology, send_counts, cost_model,
            all_results, use_llm, llm_model,
            num_nodes, max_rounds,
            unsupported_primitives=agent_sim.config.unsupported_primitives,
            op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead)

    # ---- Final ranking before hardware ----
    all_results.sort(key=lambda x: x[1]["cost_score"])
    print("\n" + "=" * 70)
    print("RANKING (post-evolution, pre-hardware)")
    print("=" * 70)
    if is_alltoallv:
        print_results(all_results, top_n=15)
    else:
        for i, (name, m) in enumerate(all_results[:15]):
            print(f"  {i+1}. {name}: {m['sim_time_us']:.1f} us, "
                  f"{m.get('local_ops', '?')} ops")

    winner_name, winner_m = all_results[0]
    print(f"\nSimulator winner: {winner_name} "
          f"(sim={winner_m['sim_time_us']:.1f} us)")

    # ---- Phase 4: Hardware evaluation ----
    hw_results = None
    if hw_eval:
        if is_alltoallv:
            hw_results, agent_sim = phase4_hardware_eval(
                all_results, send_counts, topology, agent_sim,
                use_llm, llm_model, num_nodes, worker_addrs)
        else:
            hw_results = phase4_generic_hardware_eval(
                problem, all_results, topology, num_nodes, worker_addrs,
                use_llm=use_llm, llm_model=llm_model)

    # ---- Phase 5: Code generation ----
    if output_dir:
        if is_alltoallv:
            phase5_codegen(all_results, send_counts, topology, num_nodes,
                           output_dir, hw_results=hw_results)
        else:
            phase5_generic_codegen(problem, all_results, topology, num_nodes,
                                   output_dir, hw_results=hw_results)

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
                            op_costs=None):
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
        bench = benchmark_xla_candidate_generic(
            problem, fn, topology, send_counts, world, num_nodes=num_nodes,
            unsupported_primitives=unsupported_primitives,
            op_costs=op_costs, dispatch_overhead_us=dispatch_overhead)
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

def phase3_generic_evolution(problem, topology, send_counts, cost_model,
                             baseline_results, use_llm, llm_model,
                             num_nodes, max_rounds, verbose=True,
                             unsupported_primitives=None,
                             op_costs=None, dispatch_overhead_us=100.0):
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
            dispatch_overhead_us=dispatch_overhead_us)
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
# Phase 3 (KISS Sorcar): Ablation — replace LLM evolution with Sorcar agent
# ================================================================

def phase3_kiss_sorcar(problem, topology, send_counts, cost_model,
                       baseline_results, num_nodes, verbose=True,
                       unsupported_primitives=None):
    """
    Replace Phase 3 LLM evolution with a single KISS Sorcar Buggy invocation.

    Gives Sorcar the best baseline template from Phase 2 and the benchmark
    command. Sorcar iteratively optimizes the code by running the benchmark,
    reading metrics, and editing the candidate — no islands, no scaffolding.

    Returns:
        all_results: baseline_results + sorcar result, ranked
    """
    import subprocess
    import importlib.util

    print("\n" + "=" * 70)
    print(f"[Phase 3 — KISS Sorcar Ablation] {problem.display_name}")
    print("=" * 70)

    all_results = list(baseline_results)
    project_root = Path(__file__).resolve().parent.parent

    best_name = baseline_results[0][0] if baseline_results else "unknown"
    best_template_key = list(problem.builtin_templates.keys())[0]
    for name, m in baseline_results:
        tkey = name.split(":")[-1] if ":" in name else name
        if tkey in problem.builtin_templates:
            best_template_key = tkey
            break

    template_code = problem.builtin_templates[best_template_key]

    workdir = project_root / ".kiss_sorcar_workdirs" / problem.name
    workdir.mkdir(parents=True, exist_ok=True)

    candidate_path = workdir / "candidate.py"
    candidate_path.write_text(template_code)

    python_bin = sys.executable
    bench_cmd = (
        f"{python_bin} {project_root}/experiments/kiss_sorcar_bench.py "
        f"--problem {problem.name} "
        f"--candidate {candidate_path} "
        f"--num-nodes {num_nodes}"
    )

    constraint_note = ""
    if unsupported_primitives:
        constraint_note = (
            f" IMPORTANT CONSTRAINT: The following XLA primitives do NOT "
            f"compile on the target hardware (Neuron compiler rejects them): "
            f"{unsupported_primitives}. Do NOT use these in your solution. "
            f"Use all_gather + local extraction instead."
        )

    sorcar_prompt = (
        f"Can you run the command `{bench_cmd}` in the background so that "
        f"you can monitor the output in real time, and correct the code in "
        f"the working directory if needed? If you observe any repeated "
        f"errors in the output or the command is not able to complete "
        f"successfully, please fix the code and run the command again. "
        f"Repeat the process until the command can finish successfully. "
        f"Run the command again and monitor its output in real time. You "
        f"can add diagnostic code which will print metrics, such as "
        f"running time and cost, information at finer level of granularity. "
        f"Check for opportunities to optimize the code on the basis of the "
        f"metrics information---you need to minimize the metrics. If you "
        f"discover any opportunities to minimize the metrics based on the "
        f"code and the command output, optimize the code and run the "
        f"command again. Note down the ideas you used to optimize the code "
        f"and the metrics you achieved in a file, so that you can use the "
        f"file to not repeat ideas that have already been tried and failed. "
        f"You can also use the file to combine ideas that have been "
        f"successful in the past. Repeat the process. Do not forget to "
        f"remove the diagnostic code after the optimization is complete."
        f"{constraint_note}"
    )

    prompt_file = workdir / "sorcar_prompt.txt"
    prompt_file.write_text(sorcar_prompt)

    print(f"  Seed: {best_template_key} ({best_name})")
    print(f"  Work dir: {workdir}")
    print(f"  Bench cmd: {bench_cmd}")
    print(f"  Invoking sorcar (single run, $5 budget, 600s timeout)...")

    try:
        result = subprocess.run(
            ["sorcar", "-n", "--no-web", "-w", str(workdir),
             "-f", str(prompt_file), "-b", "5"],
            capture_output=True, text=True, timeout=600,
            cwd=str(project_root))

        log_path = workdir / "sorcar_output.log"
        log_path.write_text(
            f"=== STDOUT ===\n{result.stdout}\n"
            f"=== STDERR ===\n{result.stderr}\n"
            f"=== EXIT CODE: {result.returncode} ===\n")
        print(f"  Sorcar finished (exit {result.returncode})")

    except subprocess.TimeoutExpired:
        print(f"  Sorcar timed out (600s limit)")
    except FileNotFoundError:
        print(f"  ERROR: 'sorcar' command not found")
        return all_results

    if candidate_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "sorcar_candidate", str(candidate_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, problem.evolved_fn_name, None)
            if fn is None:
                print(f"  WARNING: candidate missing {problem.evolved_fn_name}")
                return all_results

            from search.correctness_test import (
                test_xla_candidate_generic,
                benchmark_xla_candidate_generic,
            )

            ok, msg = test_xla_candidate_generic(
                problem, fn, num_nodes=num_nodes,
                unsupported_primitives=unsupported_primitives)
            if not ok:
                print(f"  WARNING: Sorcar output fails correctness: {msg}")
                return all_results

            bench = benchmark_xla_candidate_generic(
                problem, fn, topology, send_counts,
                topology.num_cores, num_nodes=num_nodes,
                unsupported_primitives=unsupported_primitives)

            if "error" in bench:
                print(f"  WARNING: Benchmark error: {bench['error']}")
                return all_results

            sim_us = bench["sim_time_us"]
            print(f"  Result: {sim_us:.1f} us, "
                  f"{bench.get('local_ops', '?')} ops, "
                  f"{bench.get('num_all_gather', 0)} ag, "
                  f"{bench.get('num_reduce_scatter', 0)} rs")

            evolved_code = candidate_path.read_text()
            all_results.append((f"sorcar:{problem.name}", {
                "template": f"sorcar_{best_template_key}",
                "cost_score": sim_us / 100.0,
                "sim_time_us": sim_us,
                "local_ops": bench.get("local_ops", "?"),
                "num_collective_permute": bench.get("num_collective_permute", 0),
                "num_all_gather": bench.get("num_all_gather", 0),
                "num_all_reduce": bench.get("num_all_reduce", 0),
                "_params": {"evolved_code": evolved_code},
            }))

        except Exception as e:
            print(f"  WARNING: Failed to evaluate Sorcar output: {e}")

    all_results.sort(key=lambda x: x[1]["cost_score"])

    print(f"\n  Sorcar complete. {len(all_results)} total candidates.")
    if all_results:
        w = all_results[0]
        print(f"  Best: {w[0]} ({w[1]['sim_time_us']:.1f} us, "
              f"{w[1].get('local_ops', '?')} local ops)")

    return all_results


# ================================================================
# Phase 5 (Generic): Code Generation → runtime/trainium_<problem>.py
# ================================================================

def phase5_generic_codegen(problem, all_results, topology, num_nodes,
                           output_dir, hw_results=None, verbose=True):
    """
    Generate final runtime code from the best candidate for any problem.
    """
    print("\n" + "=" * 70)
    print(f"[Phase 5] Code Generation → runtime/{problem.runtime_module_name}.py")
    print("=" * 70)

    world = topology.num_cores

    winner_name, winner_m = all_results[0]
    if hw_results:
        hw_sorted = sorted(
            [(name, hw_ms) for name, hw_ms, _ in hw_results if hw_ms],
            key=lambda x: x[1])
        if len(hw_sorted) >= 2:
            median_hw = sorted(h[1] for h in hw_sorted)[len(hw_sorted) // 2]
            hw_sorted = [(n, h) for n, h in hw_sorted
                         if h > median_hw / 5.0]
        if hw_sorted:
            hw_winner_name = hw_sorted[0][0]
            for name, m in all_results:
                if name == hw_winner_name:
                    winner_name, winner_m = name, m
                    print(f"  Winner selected by HW benchmark: {winner_name}")
                    break
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
    runtime_path = runtime_dir / f"{problem.runtime_module_name}.py"
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
    parser.add_argument("--kiss-sorcar", action="store_true",
                        help="Replace Phase 3 LLM evolution with KISS Sorcar agent (ablation)")
    args = parser.parse_args()

    patterns = (["moe", "skewed", "sparse", "random", "increasing", "locality"]
                if args.all_patterns else [args.pattern])

    _worker_addrs = ([a.strip() for a in args.worker_addrs.split(",")
                       if a.strip()] if args.worker_addrs else None)

    for pattern in patterns:
        run_search(
            pattern=pattern,
            use_llm=not args.no_llm,
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
            kiss_sorcar=args.kiss_sorcar,
        )
        print()


if __name__ == "__main__":
    main()
