# AllToAllV Schedule Design Prompt

You are an expert in distributed computing, network topology optimization, and collective communication algorithms on custom AI accelerator hardware.

## Hardware: AWS Trainium trn1.32xlarge

- **16 NeuronDevices**, each with **2 NeuronCores** (32 cores total)
- Devices connected in a **2D 4x4 torus** via NeuronLink
- Each device has **4 bidirectional links** (~192 GB/s per link)
- Intra-device (core-to-core on same device) is essentially free (shared HBM)
- Device adjacency (from neuron-ls):
```
Device 0:  [12, 3, 4, 1]    Device 8:  [4, 11, 12, 9]
Device 1:  [13, 0, 5, 2]    Device 9:  [5, 8, 13, 10]
Device 2:  [14, 1, 6, 3]    Device 10: [6, 9, 14, 11]
Device 3:  [15, 2, 7, 0]    Device 11: [7, 10, 15, 8]
Device 4:  [0, 7, 8, 5]     Device 12: [8, 15, 0, 13]
Device 5:  [1, 4, 9, 6]     Device 13: [9, 12, 1, 14]
Device 6:  [2, 5, 10, 7]    Device 14: [10, 13, 2, 15]
Device 7:  [3, 6, 11, 4]    Device 15: [11, 14, 3, 12]
```

## AllToAllV Operation

AllToAllV is a personalized all-to-all exchange where each rank sends a **different amount of data** to every other rank. This is critical for Mixture-of-Experts (MoE) models where tokens are routed to different experts unevenly.

### Implementation constraint (Trainium/XLA)

On Trainium, AllToAllV is implemented using **collective_permute** steps:
- Each step defines a "distance" d
- At step d: rank r sends to rank (r+d)%32 and receives from rank (r-d)%32
- All ranks participate simultaneously in each step
- Data must be **padded to max_chunk** size for static shapes (XLA requirement)
- Need 31 steps total to complete all pairwise exchanges (distances 1..31)

### Optimization goals

1. **Minimize total latency**: Order the 31 permute steps to minimize contention
2. **Topology-aware ordering**: Prefer distances that map to nearest neighbors first
3. **Reduce contention**: Avoid steps where many flows share the same physical links
4. **Traffic-pattern adaptation**: Different send_counts distributions need different orderings
5. **Minimize padding waste**: Group similar-sized transfers together if possible

## Traffic Patterns

{traffic_pattern_description}

## Current send_counts matrix

{send_counts_matrix}

## Task

Design an optimized permute schedule (ordering of distances 1..31) for the above traffic pattern on the trn1.32xlarge topology.

IMPORTANT: Your answer MUST include exactly one Python list containing all 31 integers from 1 to 31, in your chosen order. Use this exact format:

```python
schedule = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31]
```

Key topology facts for your reference:
- d=1,31 are lowest hop (avg 0.62 hops, mostly intra-device)
- d=2,30 are ~1.25 hops
- d=7,8,23,24,25 are ~1.0-1.4 hops
- d=12,20 are highest hop (avg 3.5 hops)
- For multi-node clusters: distances that cross node boundaries are ~15x more
  expensive per byte (EFA ~12.5 GB/s vs NeuronLink ~192 GB/s). Schedule these
  distant transfers to minimize inter-node link saturation.

Explain your reasoning briefly, then provide the schedule list.

## Evaluation Criteria

Your schedule will be scored on:
- **Simulated latency** on the topology model (primary)
- **Link contention** (max flows per link per step)
- **Load balance** across ranks
- **Padding waste**

Lower total score is better.
