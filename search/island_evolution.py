"""
Multi-island LLM-guided evolution for communication schedule optimization.

Runs multiple independent populations ("islands") with different selection
pressures, then uses the LLM as an intelligent crossover operator to breed
solutions across islands. Each island optimizes a different aspect of
communication quality:

  - Latency island: minimize simulated execution time (primary objective)
  - Contention island: minimize max link contention (structural objective)
  - Hop-cost island: minimize weighted hop distance (topology objective)

Why this works better than single-population GA:
1. Different fitness pressures explore different regions of schedule space
2. Migration shares discoveries without homogenizing populations
3. LLM crossover combines solutions with structural understanding of WHY
   each parent is good (via contention profiles), not just blind recombination

Inspired by AlphaEvolve's island model but specialized for communication:
- Islands have topology-specific fitness functions
- LLM crossover receives contention profiles, not just fitness scores
- Migration includes diversity checks to prevent convergence collapse
"""

import random
import re
from pathlib import Path

from .contention_analysis import ContentionAnalyzer
from .generate_algo import _invoke_bedrock
from .profiling import profile_schedule, format_profiling_report

_CROSSOVER_PROMPT = None


def _load_crossover_prompt():
    global _CROSSOVER_PROMPT
    if _CROSSOVER_PROMPT is None:
        _CROSSOVER_PROMPT = (
            Path(__file__).parent.parent / "prompts" / "island_crossover.md"
        ).read_text()
    return _CROSSOVER_PROMPT


class Island:
    """A single island with its own population and fitness function."""

    def __init__(self, name, fitness_fn, population_size=50, elements=None):
        self.name = name
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.elements = elements or list(range(1, 32))
        self.population = []
        self.best = None
        self.best_score = float("inf")

    def seed(self, schedules=None):
        self.population = []
        if schedules:
            for s in schedules[:self.population_size]:
                self.population.append(list(s))
        while len(self.population) < self.population_size:
            perm = self.elements.copy()
            random.shuffle(perm)
            self.population.append(perm)

    def evolve_generation(self, mutation_rate=0.15):
        """One generation: evaluate, select, crossover, mutate."""
        scored = [(ind, self.fitness_fn(ind)) for ind in self.population]
        scored.sort(key=lambda x: x[1])

        if scored[0][1] < self.best_score:
            self.best_score = scored[0][1]
            self.best = list(scored[0][0])

        elite_count = max(2, self.population_size // 10)
        new_pop = [list(s[0]) for s in scored[:elite_count]]

        while len(new_pop) < self.population_size:
            t1 = min(random.sample(scored, 3), key=lambda x: x[1])
            t2 = min(random.sample(scored, 3), key=lambda x: x[1])
            child = _order_crossover(t1[0], t2[0])
            child = _mutate(child, mutation_rate)
            new_pop.append(child)

        self.population = new_pop
        return self.best_score


class IslandEvolution:
    """Multi-island evolution with LLM-guided cross-island breeding."""

    def __init__(self, topology, send_counts_matrix, cost_model,
                 analyzer=None, model="haiku"):
        self.topo = topology
        self.send_counts = send_counts_matrix
        self.cost_model = cost_model
        self.analyzer = analyzer or ContentionAnalyzer(
            topology, send_counts_matrix)
        self.model = model
        self.world = topology.num_cores

    def evolve(self, template="permute_ring", elements=None,
               generations=100, island_pop=50, migration_interval=20,
               llm_crossover_count=3, seed_schedules=None, verbose=True):
        """
        Run multi-island evolution with LLM crossover.

        Returns:
            best_schedule, best_score, history
        """
        if elements is None:
            elements = list(range(1, self.world))

        islands = self._create_islands(template, elements, island_pop)
        for island in islands:
            island.seed(seed_schedules)

        history = []
        overall_best = None
        overall_best_score = float("inf")

        for gen in range(generations):
            for island in islands:
                island.evolve_generation()

            for island in islands:
                if island.best_score < overall_best_score:
                    overall_best_score = island.best_score
                    overall_best = list(island.best)

            history.append(overall_best_score)

            if (gen + 1) % migration_interval == 0:
                if verbose:
                    scores = {isl.name: f"{isl.best_score:.3f}"
                              for isl in islands}
                    print(f"  Gen {gen + 1}: {scores} "
                          f"| best={overall_best_score:.3f}")

                self._migrate(islands)

                llm_children = self._llm_crossover_round(
                    islands, template, llm_crossover_count)
                for child in llm_children:
                    score = self._evaluate(child, template, elements)
                    if score < overall_best_score:
                        overall_best_score = score
                        overall_best = list(child)
                    random.choice(islands).population[-1] = child

        if verbose:
            print(f"  Island evolution done: best={overall_best_score:.3f}")

        return overall_best, overall_best_score, history

    def _create_islands(self, template, elements, pop_size):
        """Create islands with different selection pressures."""

        def latency_fitness(sched):
            return self._evaluate(sched, template, elements)

        def contention_fitness(sched):
            diagnosis = self.analyzer.diagnose_schedule(sched, template)
            steps = diagnosis["per_step"]
            max_cont = max(s["max_contention"] for s in steps)
            avg_cont = sum(s["max_contention"] for s in steps) / len(steps)
            return max_cont * 3.0 + avg_cont

        n_entities = (self.topo.num_devices if template == "hierarchical"
                      else self.world)

        def hop_fitness(sched):
            hop_fn = (self.topo.device_hops if template == "hierarchical"
                      else self.topo.rank_hops)
            total = 0
            for step_idx, d in enumerate(sched):
                hops = sum(
                    hop_fn(r, (r + d) % n_entities)
                    for r in range(n_entities)
                )
                total += hops * (1.0 + 0.02 * step_idx)
            return total

        return [
            Island("latency", latency_fitness, pop_size, elements),
            Island("contention", contention_fitness, pop_size, elements),
            Island("hop_cost", hop_fitness, pop_size, elements),
        ]

    def _evaluate(self, schedule, template, elements):
        if template == "permute_ring":
            params = {"schedule": schedule}
        elif template == "hierarchical":
            params = {"inter_schedule": schedule}
        else:
            params = {"schedule": schedule}
        score, _ = self.cost_model.evaluate_template(template, params)
        return score

    def _migrate(self, islands):
        """Share top solutions between islands."""
        bests = [(isl.best, isl.name) for isl in islands if isl.best]
        for island in islands:
            for best, name in bests:
                if name != island.name:
                    island.population[-1] = list(best)

    def _llm_crossover_round(self, islands, template, count):
        """LLM-guided crossover between islands."""
        children = []
        parents = []
        for island in islands:
            if island.best is None:
                continue
            diagnosis = self.analyzer.diagnose_schedule(island.best, template)
            diag_text = self.analyzer.format_diagnosis(diagnosis)

            # Add profiling data to crossover context
            try:
                if template == "permute_ring":
                    prof_params = {"schedule": island.best}
                elif template == "hierarchical":
                    prof_params = {"inter_schedule": island.best}
                else:
                    prof_params = {"schedule": island.best}
                prof = profile_schedule(
                    template, prof_params, self.send_counts, self.topo)
                diag_text += "\n\n" + format_profiling_report(prof, top_k=3)
            except Exception:
                pass

            parents.append({
                "island": island.name,
                "schedule": island.best,
                "score": island.best_score,
                "diagnosis_text": diag_text,
            })

        if len(parents) < 2:
            return children

        for _ in range(count):
            p1, p2 = random.sample(parents, 2)
            try:
                child = self._llm_crossover(p1, p2)
                if child:
                    children.append(child)
            except Exception:
                continue

        return children

    def _llm_crossover(self, parent1_info, parent2_info):
        """Use LLM to intelligently cross two schedules."""
        prompt = _load_crossover_prompt().replace(
            "{parent1_island}", parent1_info["island"]
        ).replace(
            "{parent1_schedule}", repr(parent1_info["schedule"])
        ).replace(
            "{parent1_score}", f"{parent1_info['score']:.3f}"
        ).replace(
            "{parent1_diagnosis}", parent1_info["diagnosis_text"]
        ).replace(
            "{parent2_island}", parent2_info["island"]
        ).replace(
            "{parent2_schedule}", repr(parent2_info["schedule"])
        ).replace(
            "{parent2_score}", f"{parent2_info['score']:.3f}"
        ).replace(
            "{parent2_diagnosis}", parent2_info["diagnosis_text"]
        ).replace(
            "{n_elements}", str(len(parent1_info["schedule"]))
        )

        response = _invoke_bedrock(
            prompt, model=self.model, temperature=0.8, max_tokens=2048)

        expected = set(parent1_info["schedule"])
        n = len(parent1_info["schedule"])

        patterns = [
            r"(?:child_?|new_?)?schedule\s*=\s*\[([^\]]+)\]",
            r"\[(\d+(?:\s*,\s*\d+){%d,})\]" % (n - 2),
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, response):
                try:
                    nums = [int(x.strip()) for x in match.group(1).split(",")]
                    if set(nums) == expected and len(nums) == n:
                        return nums
                except ValueError:
                    continue
        return None


def _order_crossover(parent1, parent2):
    """Order crossover (OX1) for permutations."""
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


def _mutate(individual, rate=0.15):
    """Swap + segment reversal mutation."""
    ind = list(individual)
    if random.random() < rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    if random.random() < rate * 0.5:
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j] = reversed(ind[i:j])
    return ind
