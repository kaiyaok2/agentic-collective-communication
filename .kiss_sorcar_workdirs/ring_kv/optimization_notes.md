# Ring KV Optimization Notes

## Cost model
- Each collective dispatch: 100us × 2 (fwd+bwd) = 200us
- Each non-free local op: 29us
- Free ops (view, reshape, unsqueeze, squeeze, flatten, narrow, transpose, permute, expand, contiguous, slice): 0us

## Theoretical minimum
- Problem requires gathering data from all ranks → at least 1 collective
- Minimum cost = 1 × 200us = 200us

## Approaches tried

### 1. all_gather with unsqueeze/view (baseline)
- `xm.all_gather(kv_chunk.unsqueeze(0), dim=0).view(-1)` 
- Cost: 200us (unsqueeze/view are free ops)
- Result: PASS, 200us

### 2. Direct 1D all_gather (current best) ✓
- `xm.all_gather(kv_chunk, dim=0)`
- Cost: 200us (0 local ops + 1 collective)
- Result: PASS, 200us — theoretical minimum achieved
- Cleanest code, no unnecessary operations

## Approaches NOT viable
- Hierarchical (intra-node + inter-node): 2+ collectives = 400us+, always worse
- collective_permute ring: O(world_size) collectives, way worse
- all_to_all: BANNED on target hardware (Neuron compiler rejects it)
