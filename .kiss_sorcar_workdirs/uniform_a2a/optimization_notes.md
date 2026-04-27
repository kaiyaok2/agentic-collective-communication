# Uniform A2A Optimization Notes

## Cost Model
- `local_ops * 29us + num_dispatches * 2 * 100us`
- local_ops: counted ops EXCLUDING collectives (all_gather, reduce_scatter, etc.)
- Each collective dispatch costs 200us (2x for forward + backward pass)
- `torch.tensor()`, `torch.zeros()`, etc. do NOT count as ops

## Constraint
- `all_to_all` XLA primitive is BANNED (Neuron compiler rejects it)
- Must use `all_gather + local extraction` instead

## Approaches Tried

### 1. slice_loop: all_gather(unsqueeze) + view + W slices + cat
- local_ops = W + 2 (unsqueeze + view + W slices + cat, minus all_gather)
- Cost at W=64: 66*29 + 200 = 2114us
- Very expensive due to loop

### 2. all_gather(raw 1D) + index_select ✅ OPTIMAL (current)
- local_ops = 1 (index_select only)
- Dispatches = 1 (all_gather)
- Cost = 1*29 + 1*200 = 229us
- Key insight: all_gather on 1D tensor directly, then single index_select with precomputed indices

### 3. all_to_all (banned)
- Would be local_ops=0, dispatches=1 → 200us
- 29us cheaper than approach 2, but all_to_all is banned

## Theoretical Minimum (without all_to_all)
- Need ≥1 collective dispatch = 200us
- Need ≥1 local extraction op = 29us  
- Minimum = 229us ← ACHIEVED
