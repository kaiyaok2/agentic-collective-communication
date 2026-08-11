# Kiss v12 study — session summary (2026-08-11)

## Objective

Follow up on PROMPT_V11_RESULTS.md's finding that under v11:
- kiss wins clean on 4/12 novel problems (xor_grid, gray_code, piecewise, hamming_dist)
- strat wins 2/12 on sim (sum_popcount 16 vs 29, nested_mod 5 vs 5 with kiss HW-gate-fail)

Goal: use lessons from the 2 strat wins to design v12 that closes those gaps without hardcoding, leaking, or reward hacking.

## v12 prompt changes vs v11

Two hints added at Step 6 (post-correctness rewrites):

**Hint A** (extends existing dtype hint): explicit correctness argument that Neuron HW gate may fail on float % / // — recommend int64 arithmetic then cast at return. Targets nested_mod's HW-gate reference failure noted in v11 doc.

**Hint B** (new): abstract statement that small-range integer functions have vectorized closed forms combining bitwise + arithmetic ops with broadcasting. NO code snippet, no formulas, no problem names. Targets sum_popcount's suspected sim advantage from vectorized bit-decomposition vs list-comprehension constant-fold.

## Sim results (kiss v11 vs v12, 12 problems, 2-node cluster)

| Problem | v11 sim (us) | v12 sim (us) | verdict |
|---|---|---|---|
| xor_grid_bcast | 39 | 38 | v12 wins 1.03x |
| gray_code_bcast | 29 | 29 | tied |
| piecewise_bcast | 31 | 33 | v11 wins (v11 found min(i,N-i)^2 collapse) |
| triangle_num_bcast | 3 | 3 | tied |
| popcount_bcast | 29 | 29 | tied |
| hamming_dist_bcast | 29 | 19 | v12 wins 1.53x |
| cond_xor_bcast | 29 | 29 | tied |
| sum_popcount_bcast | 29 | 29 | tied (Hint B did NOT unlock the vectorized form on this problem) |
| sign_alt_bcast | 6 | 6 | tied |
| perm_shuffle_bcast | 2 | 2 | tied |
| mod_sq_bcast | 2 | 2 | tied |
| nested_mod_bcast | 5 | 5 | tied (Hint A did NOT flip the code to int64 chain either) |

## HW gate (5 focus problems x 2 configs, 64-rank)

All 10 candidates PASS HW gate. No correctness regressions.

## Real-training (2-node 64-rank, 300 iters, ms/iter)

| Problem | v11 ms/iter | v12 ms/iter | RT verdict |
|---|---|---|---|
| xor_grid_bcast | 2.63 | 2.72 | v11 wins 3.4% |
| hamming_dist_bcast | 2.22 | 3.08 | **v11 wins 39%** |
| nested_mod_bcast | 2.10 | 2.39 | v11 wins 13.6% |
| sum_popcount_bcast | 1.74 | 3.44 | **v11 wins 97%** |

**On every problem tested at real training, v11 beats v12.**

## Root cause of the RT reversal

The v12 hints push the model toward vectorized integer arithmetic (Hint B), which reduces simulator op-count but is SLOWER on Neuron HW than v11's constant-fold list-comprehension approach.

The v11 hamming_dist code:
```python
table = [[bin(i ^ j).count('1') for j in range(N)] for i in range(N)]
return torch.tensor(table, device=x.device, dtype=x.dtype)
```
becomes a compile-time NEFF constant. The v12 arithmetic version:
```python
idx = torch.arange(N, device=x.device)
ii = idx.view(N, 1); jj = idx.view(1, N)
pc = (ii + jj) % 2
for b in (2, 4, 8):
    pc = pc + (((ii // b) + (jj // b)) % 2)
```
generates a chain of runtime tensor ops. Neuron compiles the constant faster than it executes the arithmetic chain — inverting the simulator's op-count-based prediction.

## Reward-hack audit

Every v12 candidate that won sim was audited:
- **hamming_dist v12**: bit positions (1, 2, 4, 8) derived from N=16 (4 bits) — generic size fact, not a leaked reference. Formula correct at all positions. NOT a hack.
- **xor_grid v12**: 1.03x is within noise, no structural change.

No reward hacks found. The v12 wins are honest sim optimizations that HAPPEN to lose on Neuron HW.

## Conclusion

**v12 prompt REGRESSES on real hardware.** The sim's op-count cost model does not capture that Neuron prefers compile-time constants over runtime arithmetic chains for these small-tensor position-based problems. v11 remains the operational best prompt under RT-verify.

## What would v13 need to do?

Since sim-win doesn't imply RT-win here, a prompt refinement targeting the 2 strat sim wins doesn't necessarily produce a real-hardware improvement. Options for future work:

1. Move the sim/HW gap discussion INTO the prompt: warn that vectorized bit-arithmetic can be slower than constant-fold on Neuron for small tensors — but this risks leaking backend specifics.
2. Add RT feedback to the scoring loop instead of just sim (much more expensive).
3. Skip refining Hint B; keep only Hint A (int64 correctness) as it addresses correctness not perf.

None of these clearly improves the current picture. Recommendation: keep v11 as canonical.
