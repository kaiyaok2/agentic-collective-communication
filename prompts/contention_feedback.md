# Contention-Guided Schedule Refinement

You are an expert in network topology optimization for AI accelerator communication collectives. You are iteratively refining an AllToAllV schedule on AWS Trainium (trn1.32xlarge).

## Hardware
- 16 NeuronDevices in a 2D 4x4 torus, 2 NeuronCores each (32 ranks)
- Each device has 4 bidirectional NeuronLinks (~192 GB/s each)
- Intra-device communication is free (shared HBM)
- Device adjacency:
```
  0:[12,3,4,1]    4:[0,7,8,5]    8:[4,11,12,9]   12:[8,15,0,13]
  1:[13,0,5,2]    5:[1,4,9,6]    9:[5,8,13,10]    13:[9,12,1,14]
  2:[14,1,6,3]    6:[2,5,10,7]  10:[6,9,14,11]    14:[10,13,2,15]
  3:[15,2,7,0]    7:[3,6,11,4]  11:[7,10,15,8]    15:[11,14,3,12]
```

## Algorithm
Template: **{template}**
Schedule elements: {elements_description} (a permutation of {n_elements} values)

At each step, all ranks simultaneously permute data by the given distance. Steps execute sequentially; all transfers within a step run concurrently. The schedule ordering determines which distances execute first.

## Current state

**Current schedule:** `{current_schedule}`
**Current score:** {current_score} (lower is better)

## Contention diagnosis and profiling

The simulator analyzed this schedule and found the following contention and timing profile. The profiling section shows per-step simulated latency, identifying which steps dominate total execution time.

Steps marked `[INTER-NODE]` involve cross-node EFA traffic (~12.5 GB/s per adapter, ~15x slower than NeuronLink). These are typically the most expensive and should be prioritized for optimization:

{diagnosis}

## Optimization history

{history}

## Your task

Based on the contention diagnosis above, propose a **targeted modification** to improve the schedule. You can:

1. **Reorder** the full schedule (provide a complete new permutation)
2. **Swap** two specific positions: say "swap positions X and Y"
3. **Move** a distance: say "move distance D to position P"

Focus on the specific bottlenecks identified in the diagnosis. Explain your reasoning in 2-3 sentences, then provide your proposal.

IMPORTANT: If providing a full schedule, it must be a valid permutation of the same elements. Use format:
```python
schedule = [...]
```
