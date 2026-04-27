# Trainium Collective Communication Search

LLM-guided evolutionary search for optimal collective communication algorithms
on AWS Trainium (trn1.32xlarge). Uses Claude on AWS Bedrock to autonomously
profile hardware, evolve XLA-based implementations, and validate correctness
through full MoE training with autograd backward passes.

## Four Collective Problems

| Problem | Use Case | Evolved Algorithm | HW Latency (2-node) |
|---------|----------|-------------------|---------------------|
| **AllToAllV** | Variable-length MoE dispatch/combine | all\_gather + slice + cat | 2.927 ms |
| **Uniform AllToAll** | Fixed-capacity MoE token exchange | all\_gather + slice + cat | 0.546 ms |
| **Fused ReduceScatter** | Multi-tensor gradient partition (FSDP) | single reduce\_scatter on flat tensor | — |
| **Ring Attention KV** | KV cache distribution for seq-parallel | all\_gather + view | 0.353 ms |

## Training Results

All runs: 2× trn1.32xlarge, 64 ranks, bf16, DeepSeek-MoE-Lite
(DM=2048, HEADS=16, LAYERS=12, NEXP=64, TOPK=6, EXDIM=1408, SEQLEN=256,
BSZ=1, ~406M params/rank). Step times exclude 2 XLA recompilation spikes
present in every run (steady-state n=4998 of 5000 steps).

### AllToAllV

5000-step run (5 epochs, 82M tokens):

| Backend | Algorithm | Avg Step | Throughput | Wall Clock | Final Loss |
|---------|-----------|----------|------------|------------|------------|
| **evolved** | all\_gather + slice + cat (1 dispatch) | **1,095.8 ms** | 14,722 tok/s | 92.7 min | 10.38 |
| agrs | all\_gather + transpose + reduce\_scatter (2 dispatches) | 1,596.1 ms | 10,246 tok/s | 133.3 min | 10.36 |

Evolved is **46% faster** than AG+RS at this model scale.

Step time range (steady-state): evolved 1,080–1,135 ms (±2.5%); agrs 1,584–1,620 ms (±1.1%).

Final loss ≈ ln(32768) ≈ 10.4 for both — random-data training, confirming
numerical correctness rather than convergence quality.

#### AllToAllV Forward vs Backward Overhead

Isolated collective microbenchmark (DM=128, NEXP=64, TOPK=2, NTOK=128, 100 iters, 64 ranks):

| Pass | evolved (ms) | agrs (ms) | Ratio |
|------|-------------|-----------|-------|
| Forward | 3.682 | 6.624 | 1.80× faster |
| **Backward** | **3.368** | **3.413** | **~equal (1.3%)** |
| Total fwd+bwd | 7.05 | 10.037 | 1.42× faster |

The backward pass is nearly identical because both algorithms reduce to an
all\_gather in the backward direction (backward of reduce\_scatter is
all\_gather). The full-training speedup comes almost entirely from the
forward collective.

Hardware microbenchmark (2-node, 64 ranks):

| Algorithm | HW (ms) | Strategy |
|-----------|---------|----------|
| **evolved** | **2.927** | all\_gather + slice + cat (1 dispatch) |
| ag+rs | 3.273 | all\_gather + transpose + reduce\_scatter (2 dispatches) |
| naive\_allgather | 7.925 | all\_gather + slice + cat (1 dispatch, unoptimized) |
| default\_ring | 13.771 | 63 collective\_permute steps |

### Uniform AllToAll

5000-step run (5 epochs, 82M tokens):

| Backend | Algorithm | Avg Step | Throughput | Wall Clock | Final Loss |
|---------|-----------|----------|------------|------------|------------|
| **evolved** | all\_gather + slice + cat (1 dispatch) | **1,092.7 ms** | 14,955 tok/s | 91.3 min | 10.38 |
| baseline | all\_gather + transpose + reduce\_scatter (2 dispatches) | 1,593.9 ms | 10,260 tok/s | 133.1 min | 10.39 |

Evolved is **46% faster** than AG+RS baseline.

Step time range (steady-state): evolved 1,078–1,123 ms (±2%); baseline 1,581–1,623 ms (±1.3%).

Hardware microbenchmark:

| Algorithm | HW (ms) | Strategy |
|-----------|---------|----------|
| slice\_loop | 0.546 | all\_gather + per-source slice + cat (1 dispatch) |
| ag\_flat\_extract | 0.561 | all\_gather + view-based extraction (1 dispatch) |
| **baseline ag+rs** | **0.497** | all\_gather + transpose + reduce\_scatter (2 dispatches) |

Note: AG+RS wins the HW microbenchmark by ~10% at ws=64 due to optimized
reduce\_scatter kernel, but the training results show slice+cat is 46% faster
end-to-end — indicating that the ~90µs dispatch overhead and HLO graph
complexity of reduce\_scatter cost more under full model workload.

### Fused ReduceScatter

5000-step run (5 epochs, 82M tokens):

| Backend | Algorithm | Avg Step | Throughput | Wall Clock | Final Loss |
|---------|-----------|----------|------------|------------|------------|
| **baseline** | N separate reduce\_scatters (8 dispatches) | **436.9 ms** | 37,232 tok/s | 36.7 min | 10.39 |
| evolved | single reduce\_scatter on flat tensor (1 dispatch) | 427.5 ms | 35,335 tok/s | 38.6 min | 10.39 |

Baseline wall clock is 5% faster despite evolved having a lower per-step
average. The evolved run had two severe XLA recompilation spikes (max 97 s)
vs the baseline's milder spikes (max 7.4 s), costing ~2 min total. At this
tensor size the NeuronX runtime pipelines 8 independent smaller RS operations
more efficiently than one large one.

### Ring Attention KV

5000-step run (5 epochs, 82M tokens):

| Backend | Algorithm | Avg Step | Throughput | Wall Clock | Final Loss |
|---------|-----------|----------|------------|------------|------------|
| **evolved** | all\_gather + view (1 dispatch) | **667.1 ms** | 12,240 tok/s | 55.8 min | 10.39 |
| baseline | per-head all\_gather (16 dispatches per K/V) | 680.8 ms | 11,995 tok/s | 56.9 min | 10.32 |

Evolved is 2% faster. All variants are close — the KV gather is a small
fraction of total step time in this model configuration.

Step time range (steady-state): evolved 653–700 ms (±3.5%); baseline 666–713 ms (±3.4%).

## Runtime Modules

Drop-in modules written to `runtime/`:

```python
# AllToAllV (variable-length MoE dispatch)
from runtime.trainium_alltoallv import alltoallv, init_alltoallv
init_alltoallv()
output = alltoallv(x, world_size, max_chunk)

# AllToAllV with variable send/recv counts
from runtime.trainium_alltoallv import all_to_allv, init_alltoallv
output = all_to_allv(x, send_counts)

# Uniform AllToAll (fixed-capacity MoE token exchange)
from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
init_uniform_a2a()
recv = uniform_a2a(send_buf, chunk_size)

# Fused ReduceScatter (multi-tensor gradient partition)
from runtime.trainium_fused_reducescatter import fused_reducescatter, init_fused_reducescatter
init_fused_reducescatter()
shards = fused_reducescatter(gradient_tensors)

# Ring Attention KV Distribution
from runtime.trainium_ring_kv import ring_kv_gather, init_ring_kv
init_ring_kv()
all_kv = ring_kv_gather(my_kv_chunk)
```

AllToAllV AG+RS baseline is available separately:

```python
from runtime.ag_reduce_scatter import alltoallv, init_alltoallv
```

## How It Works

The search pipeline runs 5 phases for every collective problem:

```
Phase 1: Hardware Profiling — LLM agent probes hardware via tool calls,
         discovers topology, measures per-op costs, builds its own simulator.

Phase 2: Baseline Evaluation — evaluates builtin templates on the simulator,
         builds a knowledgebase of what patterns work.

Phase 3: Multi-Island Evolution — 3+ independent islands, each starting from
         a different baseline. LLM synthesizes code → sandbox → correctness
         test (including bf16) → benchmark. Per-op cost feedback each round.

Phase 4: Hardware Validation — top candidates benchmarked on real Trainium,
         then run through MoE training validation (bf16, autograd, 10 steps).
         Failed candidates get LLM-assisted recovery (up to 2 fix attempts).

Phase 5: Code Generation — winner written to runtime/trainium_<problem>.py
```

### What the Agent Discovers

The agent starts from naive seeds and must independently discover:

1. **Dispatch overhead dominates** — profiling reveals ~90µs per collective
   dispatch, making single-dispatch patterns essential
2. **Metadata ops are nearly free** — view, reshape, narrow, slice cost ~0.1µs
   vs ~29µs for compute ops like index\_select, cat, zeros
3. **Minimal-dispatch strategies** — converges to 1-dispatch patterns instead
   of multi-step ring permutes (31+ dispatches)

No hardware-specific costs are hardcoded. The agent earns its knowledge through
profiling tool calls — only measured costs flow to the evolution engine.

## Hardware Benchmarks (Phase 4, 2× trn1.32xlarge, 64 ranks)

**AllToAllV** full results:

| Algorithm | HW (ms) | Training | Strategy |
|-----------|---------|----------|----------|
| **evo:permute\_ring** | **2.927** | PASSED | all\_gather + slice + cat |
| evo:allgather\_reduce\_scatter | 3.273 | PASSED | AG+RS (2 dispatches) |
| evo:naive\_allgather | 7.925 | PASSED | all\_gather + slice + cat (unoptimized) |
| fused:default | 7.865 | — | xm.all\_to\_all (single op) |
| baseline:default\_ring | 13.771 | — | 63 collective\_permute steps |

**Uniform AllToAll**:

| Algorithm | HW (ms) | Training | Strategy |
|-----------|---------|----------|----------|
| **baseline:allgather\_reduce\_scatter** | **0.497** | PASSED | AG+RS (2 dispatches) |
| evo:slice\_loop | 0.546 | PASSED | all\_gather + per-source slice + cat |
| evo:ag\_flat\_extract | 0.561 | PASSED | all\_gather + view-based extraction |
| baseline:ag\_flat\_extract | 6.015 | PASSED | all\_gather + index\_select |

**Ring Attention KV**:

| Algorithm | HW (ms) | Training | Strategy |
|-----------|---------|----------|----------|
| **evo:naive\_ring\_permute** | **0.353** | PASSED | all\_gather + view (1 dispatch) |
| evo:flat\_allgather | 0.376 | PASSED | all\_gather + view |
| baseline:flat\_allgather | 0.508 | PASSED | all\_gather + unsqueeze + view |
| baseline:naive\_ring\_permute | 9.276 | FAILED | 63 collective\_permute steps |

## Running the Agent

```bash
# Run search for any collective problem
python experiments/run_search.py --problem alltoallv --pattern moe
python experiments/run_search.py --problem uniform_a2a
python experiments/run_search.py --problem fused_reducescatter
python experiments/run_search.py --problem ring_kv

# With hardware evaluation on 2 nodes
python experiments/run_search.py --problem alltoallv --hw-eval \
    --num-nodes 2 --master-addr 172.31.48.122 --worker-addrs 172.31.55.245

# Control evolution
python experiments/run_search.py --problem ring_kv --max-rounds 12 --llm-model sonnet

# Search without LLM (heuristics + GA + SA only)
python experiments/run_search.py --pattern moe --no-llm
```

## Running Training

```bash
# AllToAllV: evolved vs AG+RS baseline
torchrun --nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER:29500 training/train.py --backend evolved --steps 5000
torchrun --nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER:29500 training/train.py --backend agrs --steps 5000

# Uniform AllToAll
torchrun ... training/train_uniform_a2a.py --backend evolved --steps 5000

# Fused ReduceScatter
torchrun ... training/train_fused_reducescatter.py --backend evolved --steps 5000

# Ring KV
torchrun ... training/train_ring_kv.py --backend evolved --steps 5000
```

## Topology

16 NeuronDevices in a 2D 4×4 torus per node, 2 NeuronCores each (32 ranks/node).
Each device has 4 bidirectional NeuronLinks (~192 GB/s). Inter-node: 8 EFA
adapters @ 12.5 GB/s each.

```
Device adjacency (from neuron-ls):
  0:[12,3,4,1]    4:[0,7,8,5]    8:[4,11,12,9]   12:[8,15,0,13]
  1:[13,0,5,2]    5:[1,4,9,6]    9:[5,8,13,10]   13:[9,12,1,14]
  2:[14,1,6,3]    6:[2,5,10,7]  10:[6,9,14,11]   14:[10,13,2,15]
  3:[15,2,7,0]    7:[3,6,11,4]  11:[7,10,15,8]   15:[11,14,3,12]
```

## Correctness Verification

Every algorithm passes 5 verification layers:

1. **Reference implementation** — pure-Python ground truth per problem
2. **Multi-rank simulation** — TrackedTensor + CollectiveSimulator across
   world sizes (4, 8) and multiple traffic patterns
3. **bf16 correctness** — candidates tested with bfloat16 inputs
4. **Sandbox execution** — restricted environment, no file I/O or imports
5. **Real hardware verification** — diagnostic inputs with post-execution checks

## Requirements

- AWS Trainium instance (trn1.32xlarge) with `neuronxcc` and `torch-neuronx`
- `boto3` for Bedrock LLM access (default: Claude Sonnet; `--no-llm` for search without LLM)
- Multi-node: 2+ trn1.32xlarge in same VPC with SSH + EFA
