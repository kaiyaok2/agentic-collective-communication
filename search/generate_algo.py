"""
LLM-based AllToAllV schedule generator.

Uses Claude on AWS Bedrock to propose topology-aware permute schedules,
combined with algorithmic search (genetic algorithm, simulated annealing)
for refinement.
"""

import json
import random
import math
import re
import itertools
from pathlib import Path

import boto3

PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "schedule_design.md").read_text()

BEDROCK_REGION = "us-east-2"
# Sonnet 4.5 for high quality; Haiku 4.5 for fast iteration
MODEL_IDS = {
    "sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus": "us.anthropic.claude-opus-4-6-v1",
}


def _invoke_bedrock(prompt, model="haiku", max_tokens=4096, temperature=1.0):
    """Call Claude via Bedrock and return text response."""
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=MODEL_IDS[model], body=body)
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def _parse_schedule(text, world=32):
    """Extract a schedule (list of ints) from LLM response text."""
    expected = set(range(1, world))

    # Try multiple regex patterns
    patterns = [
        r"schedule\s*=\s*\[([^\]]+)\]",
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                nums = [int(x.strip()) for x in match.group(1).split(",")]
                if set(nums) == expected and len(nums) == world - 1:
                    return nums
            except ValueError:
                continue

    # Fallback: extract all integers and check if there's a valid permutation
    all_nums = [int(x) for x in re.findall(r'\b(\d+)\b', text)]
    # Sliding window of size world-1
    for i in range(len(all_nums) - (world - 2)):
        candidate = all_nums[i:i + world - 1]
        if set(candidate) == expected:
            return candidate

    return None


def _format_send_counts(send_counts_matrix, world):
    """Format send_counts matrix for the prompt."""
    lines = []
    for src in range(min(world, 8)):  # Show first 8 ranks to keep prompt reasonable
        row = [send_counts_matrix[src][dst] for dst in range(world)]
        lines.append(f"  Rank {src}: {row}")
    if world > 8:
        lines.append(f"  ... ({world - 8} more ranks)")
    return "\n".join(lines)


def generate_llm_schedule(send_counts_matrix, traffic_description="general",
                          model="haiku", num_candidates=3, temperature=1.0):
    """
    Use Claude to generate candidate AllToAllV schedules.

    Returns:
        list of (schedule, reasoning) tuples
    """
    world = len(send_counts_matrix)
    prompt = PROMPT_TEMPLATE.replace(
        "{traffic_pattern_description}", traffic_description
    ).replace(
        "{send_counts_matrix}", _format_send_counts(send_counts_matrix, world)
    )

    candidates = []
    for i in range(num_candidates):
        try:
            response = _invoke_bedrock(prompt, model=model,
                                       temperature=temperature)
            schedule = _parse_schedule(response, world)
            if schedule:
                candidates.append((schedule, response))
            else:
                print(f"  [LLM candidate {i}] Failed to parse schedule from response")
        except Exception as e:
            print(f"  [LLM candidate {i}] Bedrock error: {e}")

    return candidates


# ============================================================
# Algorithmic schedule generators (no LLM needed)
# ============================================================

def topology_aware_schedule(topology, world=32):
    """
    Generate schedule ordered by hop distance (nearest first).
    Breaks ties by preferring distances that spread load across links.
    """
    hop_costs = {}
    for d in range(1, world):
        total_hops = 0
        for r in range(world):
            dst = (r + d) % world
            total_hops += topology.rank_hops(r, dst)
        hop_costs[d] = total_hops

    return sorted(range(1, world), key=lambda d: (hop_costs[d], d))


def butterfly_schedule(world=32):
    """
    Butterfly/hypercube-style schedule: powers of 2 first, then combinations.
    Good for reducing diameter-limited latency.
    """
    powers = []
    d = 1
    while d < world:
        powers.append(d)
        d *= 2

    remaining = [d for d in range(1, world) if d not in powers]
    random.shuffle(remaining)
    return powers + remaining


def contention_aware_schedule(topology, world=32):
    """
    Greedy contention-minimizing schedule.
    At each step, pick the distance that causes least contention
    given the distances already chosen.
    """
    from collections import defaultdict

    remaining = set(range(1, world))
    schedule = []

    while remaining:
        best_d = None
        best_contention = float("inf")

        for d in remaining:
            # Estimate contention for this distance
            link_usage = defaultdict(int)
            for r in range(world):
                dst = (r + d) % world
                src_dev = topology.rank_to_device(r)
                dst_dev = topology.rank_to_device(dst)
                if src_dev == dst_dev:
                    continue
                path = topology.device_path(src_dev, dst_dev)
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    link_usage[key] += 1

            max_contention = max(link_usage.values()) if link_usage else 0
            if max_contention < best_contention:
                best_contention = max_contention
                best_d = d

        schedule.append(best_d)
        remaining.remove(best_d)

    return schedule


def traffic_adaptive_schedule(topology, send_counts_matrix, world=32):
    """
    Traffic-aware schedule: prioritize distances with the most data to send.
    Within same-data-volume groups, prefer lower hop counts.
    """
    distance_volume = {}
    for d in range(1, world):
        total = 0
        for r in range(world):
            dst = (r + d) % world
            total += send_counts_matrix[r][dst]
        distance_volume[d] = total

    distance_hops = {}
    for d in range(1, world):
        total = 0
        for r in range(world):
            dst = (r + d) % world
            total += topology.rank_hops(r, dst)
        distance_hops[d] = total

    # Sort: highest volume first, then lowest hop count
    return sorted(range(1, world),
                  key=lambda d: (-distance_volume[d], distance_hops[d]))


# ============================================================
# Genetic algorithm for schedule refinement
# ============================================================

def genetic_search(cost_fn, world=32, population_size=100,
                   generations=200, mutation_rate=0.15, elite_frac=0.1,
                   seed_schedules=None, elements=None):
    """
    Genetic algorithm to optimize permute schedule ordering.

    Uses adaptive parameters: cosine-decaying mutation rate, adaptive
    tournament size, stagnation detection with random immigrants, and
    diverse mutation operators (swap, reversal, insert, scramble).

    cost_fn: callable(schedule) -> float (lower is better)
    seed_schedules: optional list of schedules to include in initial population
    elements: list of elements to permute (default: range(1, world))

    Returns:
        best_schedule, best_cost, history
    """
    distances = elements if elements is not None else list(range(1, world))

    # Initialize population
    population = []
    if seed_schedules:
        for s in seed_schedules:
            population.append(list(s))

    while len(population) < population_size:
        perm = distances.copy()
        random.shuffle(perm)
        population.append(perm)

    def evaluate(pop):
        return [(ind, cost_fn(ind)) for ind in pop]

    def crossover(parent1, parent2):
        """Order crossover (OX1)."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        fill = [x for x in parent2 if x not in child[start:end]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def mutate(individual, rate):
        """Swap + reversal + insert + scramble mutation with adaptive rate."""
        ind = individual.copy()
        n = len(ind)
        # Swap mutation
        if random.random() < rate:
            i, j = random.sample(range(n), 2)
            ind[i], ind[j] = ind[j], ind[i]
        # Segment reversal
        if random.random() < rate * 0.5:
            i, j = sorted(random.sample(range(n), 2))
            ind[i:j] = reversed(ind[i:j])
        # Insert mutation: remove element, reinsert elsewhere
        if random.random() < rate * 0.3:
            i = random.randrange(n)
            elem = ind.pop(i)
            j = random.randrange(n)
            ind.insert(j, elem)
        # Scramble mutation: shuffle a small sub-segment
        if random.random() < rate * 0.2:
            seg_len = random.randint(3, min(5, n))
            start = random.randint(0, n - seg_len)
            seg = ind[start:start + seg_len]
            random.shuffle(seg)
            ind[start:start + seg_len] = seg
        return ind

    elite_count = max(2, int(population_size * elite_frac))
    best_ever = None
    best_cost = float("inf")
    history = []
    stagnation_count = 0

    for gen in range(generations):
        # Adaptive mutation rate: cosine decay from 0.3 to 0.05
        adaptive_rate = 0.05 + 0.25 * (1 + math.cos(math.pi * gen / generations)) / 2
        # Adaptive tournament size: linearly increase from 2 to 5
        tourn_size = 2 + int(3 * gen / max(generations - 1, 1))

        scored = evaluate(population)
        scored.sort(key=lambda x: x[1])

        gen_best_cost = scored[0][1]
        history.append(gen_best_cost)

        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_ever = scored[0][0].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Selection: elitism + tournament
        elites = [s[0] for s in scored[:elite_count]]
        new_pop = [e.copy() for e in elites]

        # Stagnation detection: inject random immigrants and boost mutation
        if stagnation_count >= 20:
            num_immigrants = max(1, population_size // 10)
            for _ in range(num_immigrants):
                perm = distances.copy()
                random.shuffle(perm)
                new_pop.append(perm)
            adaptive_rate = min(adaptive_rate * 2.0, 0.5)
            stagnation_count = 0

        while len(new_pop) < population_size:
            # Adaptive tournament selection
            t1 = min(random.sample(scored, min(tourn_size, len(scored))),
                     key=lambda x: x[1])
            t2 = min(random.sample(scored, min(tourn_size, len(scored))),
                     key=lambda x: x[1])
            child = crossover(t1[0], t2[0])
            child = mutate(child, adaptive_rate)
            new_pop.append(child)

        population = new_pop

    return best_ever, best_cost, history


def local_search(cost_fn, initial_schedule, max_rounds=100):
    """
    Steepest-descent local search: enumerate all n*(n-1)/2 swaps,
    pick the best improving swap, repeat until no swap improves.

    Returns:
        best_schedule, best_cost, num_rounds
    """
    current = list(initial_schedule)
    current_cost = cost_fn(current)
    n = len(current)

    for round_idx in range(max_rounds):
        best_swap = None
        best_swap_cost = current_cost

        for i in range(n):
            for j in range(i + 1, n):
                current[i], current[j] = current[j], current[i]
                c = cost_fn(current)
                if c < best_swap_cost:
                    best_swap_cost = c
                    best_swap = (i, j)
                current[i], current[j] = current[j], current[i]

        if best_swap is None:
            break

        i, j = best_swap
        current[i], current[j] = current[j], current[i]
        current_cost = best_swap_cost

    return current, current_cost, round_idx + 1


def simulated_annealing(cost_fn, world=32, initial_schedule=None,
                        max_iters=5000, initial_temp=10.0, cooling=0.997,
                        elements=None):
    """
    Simulated annealing for schedule optimization.

    Returns:
        best_schedule, best_cost, history
    """
    if initial_schedule is None:
        current = list(elements) if elements is not None else list(range(1, world))
        random.shuffle(current)
    else:
        current = list(initial_schedule)

    current_cost = cost_fn(current)
    best = current.copy()
    best_cost = current_cost
    temp = initial_temp
    history = []

    for i in range(max_iters):
        # Neighbor: swap two random positions
        neighbor = current.copy()
        a, b = random.sample(range(len(neighbor)), 2)
        neighbor[a], neighbor[b] = neighbor[b], neighbor[a]

        neighbor_cost = cost_fn(neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

        temp *= cooling
        history.append(best_cost)

    return best, best_cost, history
