#!/usr/bin/env python3
"""
Benchmark harness for KISS Sorcar Phase 3 ablation.

Loads a candidate collective implementation from candidate.py in the
working directory, runs correctness tests and benchmarks it on the
simulator, and prints metrics.

Usage:
    python experiments/kiss_sorcar_bench.py --problem uniform_a2a

The candidate must define the function specified by the problem's signature
(e.g., evolved_uniform_a2a for uniform_a2a).
"""

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.problems import get_problem
from search.correctness_test import (
    test_xla_candidate_generic,
    benchmark_xla_candidate_generic,
)
from simulator.topology import TrainiumTopology, MultiNodeTopology


def _make_send_counts(world, shard_size=1024):
    matrix = [[0] * world for _ in range(world)]
    rng = random.Random(42)
    raw = [1.0 / (i + 1) ** 1.2 for i in range(world)]
    perm = list(range(world))
    rng.shuffle(perm)
    probs = [0.0] * world
    for i, p in enumerate(perm):
        probs[p] = raw[i]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    for src in range(world):
        for dst in range(world):
            if src != dst:
                w = probs[src] * probs[dst]
                matrix[src][dst] = max(1, int(shard_size * w * world * world))
    return matrix


def load_candidate(candidate_path, fn_name):
    spec = importlib.util.spec_from_file_location("candidate_mod", candidate_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        avail = [k for k in dir(mod) if not k.startswith("_")]
        raise RuntimeError(
            f"candidate.py does not define '{fn_name}'. "
            f"Available: {avail}")
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--candidate", default="candidate.py")
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()

    problem = get_problem(args.problem)
    candidate_path = Path(args.candidate).resolve()

    print(f"Problem: {problem.display_name}")
    print(f"Candidate: {candidate_path}")
    print(f"Function: {problem.evolved_fn_name}")
    print()

    fn = load_candidate(candidate_path, problem.evolved_fn_name)

    t0 = time.time()
    ok, msg = test_xla_candidate_generic(
        problem, fn, num_nodes=args.num_nodes, verbose=True)
    test_time = time.time() - t0

    print(f"Correctness: {'PASS' if ok else 'FAIL'} ({test_time:.1f}s)")
    if not ok:
        print(f"  Error: {msg}")
        print()
        print("METRICS: correctness=FAIL")
        sys.exit(1)

    if args.num_nodes > 1:
        topology = MultiNodeTopology(num_nodes=args.num_nodes)
    else:
        topology = TrainiumTopology()

    world = topology.num_cores
    send_counts = _make_send_counts(world)

    bench = benchmark_xla_candidate_generic(
        problem, fn, topology, send_counts, world,
        num_nodes=args.num_nodes)

    if "error" in bench:
        print(f"Benchmark error: {bench['error']}")
        print("METRICS: correctness=PASS benchmark=ERROR")
        sys.exit(1)

    sim_us = bench["sim_time_us"]
    local_ops = bench.get("local_ops", "?")
    n_ag = bench.get("num_all_gather", 0)
    n_rs = bench.get("num_reduce_scatter", 0)
    n_ar = bench.get("num_all_reduce", 0)
    n_cp = bench.get("num_collective_permute", 0)

    print()
    print(f"METRICS: correctness=PASS sim_time_us={sim_us:.1f} "
          f"local_ops={local_ops} "
          f"all_gather={n_ag} reduce_scatter={n_rs} "
          f"all_reduce={n_ar} collective_permute={n_cp} "
          f"cost_score={sim_us/100:.2f}")


if __name__ == "__main__":
    main()
