# Model-extension follow-on: TP / FSDP / PP with microbatching

This directory contains the scripts that produced the four model-extension
positive results documented in the paper's §9 follow-on paragraph and the
top-level README's "Non-OLMoE follow-on" table.

All scripts run on the same 7-node trn1.32xlarge cluster (224 ranks total)
that we use for the four headline OLMoE problems. They share the structural
pattern that the paper's headline problems exploit *within a single
mark_step graph* (XLA fuses adjacent collectives) and put it *across*
mark_step boundaries by forcing a per-microbatch barrier — which is the
naive shape a developer writes for memory-pressure-driven microbatching,
gradient accumulation, or pipeline scheduling.

## Scripts

| Script | What it tests | per_mb / bundled / speedup |
|---|---|---|
| `train_pp_llama.py` | Llama-style 2-stage pipeline-parallel forward with masked-AR cross-stage transfer; M microbatches | M=2: 11.12/9.16/1.21x; M=4: 21.17/14.15/**1.50x**; M=8: 34.78/25.97/**1.34x** |
| `train_pp_llama_bwd.py` | Same as above but with full autograd backward (masked AR has identity backward) | M=4: 37.60/10.57/**3.56x** |
| `train_tp_mb.py` | Llama-style TP head/MLP parallelism: column+row-parallel SwiGLU MLP, one AR per layer, 4 layers | M=4: 7.06/4.49/**1.57x** |
| `train_fsdp_mb.py` | FSDP-style sharded weight prefetch: per-layer weight AG, 4 layers | M=4: 4.40/3.03/**1.45x** |

## Common structural shape

In every test the baseline (`per_mb`) issues `xm.mark_step()` between
microbatches — this is the natural code a developer writes when
microbatching is real (memory pressure, gradient accumulation, pipeline
schedule). Neuron's XLA cannot fuse collectives across mark_steps, so M
microbatches each containing K collectives produce M*K structurally
unfused HLO graphs (XLA still fuses the K collectives inside each
microbatch's graph into 1).

The agent (`bundled`) variant puts all M microbatches' work in a single
mark_step graph; XLA then fuses everything to ~1 collective per
primitive type. Trainium has no async collectives, so the bundled
variant loses no pipelining benefit; on this stack pipelining was never
overlapped with compute anyway.

## What's still needed for paper integration

These are real-training measurements but they are not yet integrated
into the search loop. For each direction:

1. Add a `CollectiveProblem` entry in `search/problems.py` with a
   pure-Python reference, seed templates (`per_mb` and `bundled`), and
   test-case generator.
2. Extend the simulator's `T_launch` term to model per-mb mark_step
   boundaries explicitly (currently the cost model amortises across
   one `autograd.Function` boundary; for microbatching it should be
   M dispatch costs not 2).
3. Run the multi-island evolution loop on the new problem to confirm
   the agent independently rediscovers the bundled composition.

The expected output of step 3 is a `runtime/trainium_pp_send_recv.py`
(or analogous) module that drops into a real training loop.

## How to run

From the repo root with the cluster up:

```bash
PORT=33700 bash <your runner script for these>  # see /tmp/tp_search/run_pp.sh
```

The repository's existing 7-node bench runners under
`experiments/h7_bench/` use the same launching pattern. Adapt those.
