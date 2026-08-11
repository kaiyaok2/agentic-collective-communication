# Real-training results (2-node, 64-rank, 300 iters)

| Problem | v11 ms/iter | v12 ms/iter | v11/v12 |
|---|---|---|---|
| xor_grid_bcast | 2.6330 | 2.7224 | 1.03x v11 |
| hamming_dist_bcast | 2.2188 | 3.0801 | 1.39x v11 |
| nested_mod_bcast | 2.1025 | 2.3882 | 1.14x v11 |
| sum_popcount_bcast | 1.7435 | 3.4393 | 1.97x v11 |

Sim (kiss v12) wins on hamming_dist_bcast 1.53x, but real training reverses to a 1.39x v11 win.
Cause: sim underweights runtime cost of arithmetic chains vs compile-time constant tensors on Neuron.
