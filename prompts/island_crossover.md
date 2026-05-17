# Cross-Island Schedule Breeding

You are combining two AllToAllV schedules from different optimization islands. Each island optimizes a different objective. Your goal: create a child schedule that inherits the best properties of both parents.

## Parent 1 (from "{parent1_island}" island)
**Schedule:** `{parent1_schedule}`
**Score:** {parent1_score}

Contention profile:
{parent1_diagnosis}

## Parent 2 (from "{parent2_island}" island)
**Schedule:** `{parent2_schedule}`
**Score:** {parent2_score}

Contention profile:
{parent2_diagnosis}

## Your task

Analyze what makes each parent good at its island's objective, then design a child schedule that combines:
- The low-contention ordering patterns from the contention-optimized parent
- The low-latency step placement from the latency-optimized parent
- The topology-aware distance grouping from the hop-cost parent
- For multi-node clusters: inter-node steps (marked [INTER-NODE]) dominate latency.
  Prioritize inheriting parent placements that minimize EFA adapter saturation.

Think about which distances are placed well in each parent and why, then construct a child that inherits the best placements.

IMPORTANT: The child must be a valid permutation of exactly {n_elements} elements (same elements as the parents). Use format:
```python
schedule = [...]
```

Explain your reasoning in 2-3 sentences, then provide the child schedule.
