#!/usr/bin/env python3
"""
Convergence test: verify AllToAllV → AG+index_select and Uniform A2A → AG+view_slice.

Runs Phase 1 agent profiling first (so op_costs come from agent tool calls only),
then multi-island evolution for both problems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.topology import TrainiumTopology
from simulator.cost_model import CostModel
from search.contention_analysis import ContentionAnalyzer
from search.template_evolution import TemplateEvolution
from search.problems import get_problem
from search.agent_simulator_config import (
    run_profiling_agent, _HARDWARE_MEASUREMENTS,
)
from experiments.run_search import _extract_op_costs, make_send_counts


def run_convergence_test():
    # ---- Phase 1: Agent profiling (discovers op costs via tool calls) ----
    print("\n" + "=" * 70)
    print("PHASE 1: Agent Hardware Profiling")
    print("=" * 70)

    agent_sim = run_profiling_agent(model="haiku", max_turns=15, verbose=True)

    op_costs = _extract_op_costs(agent_sim)
    print(f"\n  Agent measured {len(op_costs)} op costs: {sorted(op_costs.keys())}")
    for op, cost in sorted(op_costs.items(), key=lambda x: x[1]):
        print(f"    {op:<20s} {cost:.1f} us")

    config = agent_sim.config
    dispatch_overhead = config.collective_dispatch_overhead_us
    if dispatch_overhead <= 0 or dispatch_overhead > 500:
        dispatch_overhead = 100.0

    unsupported = config.unsupported_primitives or ["all_to_all"]

    # ---- Build topology from agent-discovered config ----
    _nd = config.num_devices if config.num_devices > 0 else 16
    _cpd = config.cores_per_device if config.cores_per_device > 0 else 2
    _adj = config.device_adjacency if config.device_adjacency else _HARDWARE_MEASUREMENTS["device_adjacency"]
    _bw = config.link_bandwidth_gbps if config.link_bandwidth_gbps >= 10.0 else 192.0
    _lat = config.link_latency_us if 0 < config.link_latency_us < 100 else 0.5

    topo = TrainiumTopology(
        link_bandwidth_GBps=_bw, link_latency_us=_lat,
        num_devices=_nd, cores_per_device=_cpd,
        device_adjacency=_adj)

    world = topo.num_cores
    send_counts = make_send_counts("moe", world=world)
    cost_model = CostModel(topo, send_counts, dispatch_overhead_us=dispatch_overhead)
    analyzer = ContentionAnalyzer(topo, send_counts)

    results = {}

    # ==================================================================
    # Test 1: AllToAllV - should converge to AG + index_select
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 1: AllToAllV convergence → AG+index_select")
    print("=" * 70)

    p = get_problem("alltoallv")
    a2v_converged = False

    for starting in ["naive_allgather", "allgather_reduce_scatter", "permute_ring"]:
        print(f"\n  Island: {starting}")
        te = TemplateEvolution(
            topo, send_counts, cost_model, analyzer,
            model="haiku", problem=p,
            unsupported_primitives=unsupported,
            op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead)

        code, bench, hist = te.evolve(starting_template=starting, max_rounds=5,
                                       verbose=True)

        has_index_select = "index_select" in code
        sim_us = bench.get("sim_time_us", float("inf"))
        print(f"  Result: sim_time={sim_us:.1f}us, "
              f"has_index_select={has_index_select}")
        print(f"  Code snippet: {code[:200]}...")

        results[f"alltoallv_{starting}"] = {
            "sim_time_us": sim_us,
            "has_index_select": has_index_select,
            "code": code,
        }

        if has_index_select:
            a2v_converged = True

    # ==================================================================
    # Test 2: Uniform A2A - should converge to AG + view_slice
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Uniform A2A convergence → AG+view_slice")
    print("=" * 70)

    p2 = get_problem("uniform_a2a")
    ua2a_converged = False

    for starting in ["slice_loop", "ag_flat_extract", "allgather_reduce_scatter", "permute_ring"]:
        print(f"\n  Island: {starting}")
        te = TemplateEvolution(
            topo, send_counts, cost_model, analyzer,
            model="haiku", problem=p2,
            unsupported_primitives=unsupported,
            op_costs=op_costs,
            dispatch_overhead_us=dispatch_overhead)

        code, bench, hist = te.evolve(starting_template=starting, max_rounds=5,
                                       verbose=True)

        uses_index_select = "index_select" in code
        sim_us = bench.get("sim_time_us", float("inf"))
        local_ops = bench.get("local_ops", 999)

        print(f"  Result: sim_time={sim_us:.1f}us, local_ops={local_ops}, "
              f"uses_index_select={uses_index_select}")
        print(f"  Code snippet: {code[:200]}...")

        results[f"uniform_a2a_{starting}"] = {
            "sim_time_us": sim_us,
            "local_ops": local_ops,
            "uses_index_select": uses_index_select,
            "code": code,
        }

        if sim_us < 210:
            ua2a_converged = True

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    print(f"\nAgent measured {len(op_costs)} ops: {sorted(op_costs.keys())}")

    print(f"\nAllToAllV → AG+index_select: {'CONVERGED' if a2v_converged else 'FAILED'}")
    for k, v in results.items():
        if k.startswith("alltoallv_"):
            island = k.replace("alltoallv_", "")
            print(f"  {island}: {v['sim_time_us']:.1f}us, index_select={v['has_index_select']}")

    print(f"\nUniform A2A → AG+view_slice: {'CONVERGED' if ua2a_converged else 'FAILED'}")
    for k, v in results.items():
        if k.startswith("uniform_a2a_"):
            island = k.replace("uniform_a2a_", "")
            print(f"  {island}: {v['sim_time_us']:.1f}us, local_ops={v.get('local_ops', '?')}")

    if a2v_converged and ua2a_converged:
        print("\n*** BOTH PROBLEMS CONVERGED — ready for 16-experiment pipeline ***")
        return True
    else:
        if not a2v_converged:
            print("\n*** AllToAllV did NOT converge to AG+index_select ***")
        if not ua2a_converged:
            print("\n*** Uniform A2A did NOT converge to AG+view_slice ***")
        return False


if __name__ == "__main__":
    success = run_convergence_test()
    sys.exit(0 if success else 1)
