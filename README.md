# Composing Collectives Above a Black-Box Vendor Library on AWS Trainium

Profiling-driven search for fast composite collectives on AWS Trainium
(trn1.32xlarge). The vendor's collective library is treated as an oracle
— we never modify it and never see inside it; we only call its primitives
(`xm.all_gather`, `xm.reduce_scatter`, `xm.all_reduce`, `xm.collective_permute`,
`xm.all_to_all`) and pre/post-process the data with local XLA ops.

Within that layer the agent searches over compositions: how many primitive
dispatches, in what order, against what tensor layout, with what local
work in between. A simulator built from on-device profiling ranks
candidates; an LLM proposes new candidates; the top survivors are
benchmarked on real Trainium and validated end-to-end in DeepSeek-MoE-Lite
training. Search results are emitted as `runtime/trainium_<problem>.py`
modules that drop into a real training loop.

## Why this framing, and what it is *not*

This is not a competitor to MSCCL/TACCL, AlphaEvolve, or FunSearch.

| Prior work | What they assume / target | Why they don't apply here |
|---|---|---|
| **MSCCL / TACCL** | Programmable schedules, open primitives (NCCL/MSCCL-IR), explicit chunking and permutation steps | Trainium's collective library is closed; we cannot rewrite the schedule below the `xm.*` boundary |
| **AlphaEvolve / FunSearch** | Search the algorithm space — invent new algorithms (matrix multiplication, hashing, sort) | We're not looking for new algorithms; we're looking for the right way to *compose* a small fixed primitive set against a black-box backend |
| **AutoTVM / Triton autotune** | Search the kernel implementation space (loop tiling, vectorization, schedule) | We never write kernels; the backend writes the kernel for each primitive call |

The contribution is a *closed-stack* search loop: an empirical, on-device
profiling layer + a cost model that captures vendor-library quirks
(per-graph NEFF compilation cost, per-mark_step overhead, per-op
kernel-launch floor, contiguity-aware implicit-copy charging) + an LLM
that proposes compositions, all gated through correctness tests against
a pure-Python reference and an end-to-end MoE training validation step.
Reviewers cannot ask "why LLM vs. classical search" because the LLM is
not the contribution — the *loop* is. They cannot ask "why not compare
to MSCCL" because MSCCL operates at a layer we can't access on this
hardware. The only honest comparison is *against the developer-written
baseline a practitioner would write inside a real training script*,
which is what we report.

## Five Collective Problems

| Problem | Use Case | Agent's Composition |
|---|---|---|
| **AllToAllV** | Variable-length MoE dispatch/combine | pack + 1 `all_gather` + `index_select` |
| **Uniform AllToAll** | Fixed-capacity MoE token exchange | 1 `all_gather` + per-source slice + `cat` |
| **Ring Attention KV** | KV cache distribution for seq-parallel | one `all_gather` per slot (K, V) — 2 dispatches per call |
| **Gradient AllReduce** | Sync of replicated (non-expert) gradients each step | concat all replicated grads + 1 fused `all_reduce` + split |
| **Distributed Cross-Entropy** | Loss on vocab-sharded logits without materializing full `(B·S, V)` | 2 `all_reduce`s: one on local `Σ exp(logits)`, one on per-token target logit (bf16-safe, no max-shift) |

## Per-Problem MoE-Style Microbenchmark Training (1-Node)

**Each `train_*.py` is a single-collective microbenchmark, not real
end-to-end DeepSeek-V2-Lite training.** The harness wraps a MoE-shaped
transformer around *one* collective pattern at realistic shapes and
runs it under `NEURON_NUM_RECENT_MODELS_TO_KEEP=1` graph-cache
pressure. The 5000-step number is the per-step training cost of that
one collective composition in a cache-pressured loop. Loss values
floor at `log(VOCAB) ≈ 10.39` because the dataset is random tokens —
the model doesn't actually learn.

Per script, what's MoE-shaped vs. what's purely synthetic:
- `train.py`, `train_uniform_a2a.py` — shared MoE shell (attention +
  MoE block), 12 layers.
- `train_ring_kv.py` — same MoE shell, but the attention layer is
  *replaced* with a `SeqParallelAttn` doing ring-KV distribution. Real
  DeepSeek-V2-Lite uses MLA, not ring-KV; this script exists to time
  the ring-KV pattern at realistic shapes.

For an end-to-end run of the AllToAllV collective inside a real MoE
training loop, see the **7-Node OLMoE-Style End-to-End Comparison**
section below.

Single-node, **1× trn1.32xlarge (32 ranks)**, 5000 steps, bf16,
full autograd backward, `NEURON_NUM_RECENT_MODELS_TO_KEEP=1`.

Model config: `DM=2048, HEADS=16, LAYERS=12 (8 for Ring KV), NEXP=64,
TOPK=6, EXDIM=1408, SEQLEN=256 (128 for Ring KV), BSZ=1`. The agent
runtime is the file `runtime/trainium_<problem>.py` produced by the
search; the baseline is the developer-written composition from the
project training scripts. Step times are end-to-end wall-clock per step
including forward, backward, optimizer, and amortized NEFF cache reload.

Each of the three problem sections below reports two metrics:

- **Per-call HW (ms)** — single-call latency from the Phase 4 hardware
  microbench (20-iter, fresh tensors, fixed shape sized for the
  microbench, world=32). Captures isolated dispatch + framework cost
  but *not* NEFF cache eviction or per-mark_step overhead at the
  thousands-of-graphs scale that real training induces.
- **5000-step avg (ms)** — full DeepSeek-MoE-Lite per-step time, bf16,
  full autograd backward, `NEURON_NUM_RECENT_MODELS_TO_KEEP=1`. The
  number that matters for actual training. Final-loss reported only
  for the agent's pick + the baseline used in the speedup comparison.

### AllToAllV

| Backend | Composition | Per-call HW (ms) | 5000-step avg | Wall Clock | Final Loss |
|---|---|---:|---:|---:|---:|
| **agent** | pack + 1 `all_gather` + `index_select` | 2.75 | **655.1 ms** | 54.6 min | 10.37 |
| baseline (`agrs`) | `all_gather` + `reduce_scatter` (2 dispatches) | 1.66 | 816.0 ms | 68.1 min | 10.39 |

Agent is **1.246× faster** than the AG+RS baseline at 5000-step
training scale. The agent's pack layout (rank `i`'s data at
`[i*max_chunk]` in a fixed-size buffer) lets the entire variable-length
operation reduce to one `all_gather` plus an `index_select` for
receive-side extraction.

Per-call HW microbench prefers the AG+RS baseline (1.66 ms vs 2.75 ms
for the agent) because in isolation, with NEFFs already warm in the
cache from the warmup iteration, the agent's pack-buffer construction
and `index_select` add local work that the AG+RS path avoids. At
5000-step training scale the order flips: each `_A2AV.forward`
mark_step pair compiles a fresh HLO graph, AG+RS produces a graph
with two collectives where the agent's produces one, and the combined
AG+RS graph drives a larger NEFF that pays more cache reload events
under `NEURON_NUM_RECENT_MODELS_TO_KEEP=1`. The simulator captures
this via `compilation_cost(largest_collective_tensor)` amortized over
training steps; HW microbench does not.

### Uniform AllToAll

| Backend | Composition | Per-call HW (1-node ms) | Per-call training (7-node ms) | 5000-step avg | 1000-step steady (7-node) | Final Loss |
|---|---|---:|---:|---:|---:|---:|
| **agent** | 1 `all_gather` + per-source slice + `cat` | **3.67** | **66.15** | **655.8 ms** | **2117.6 ms** | 10.38 |
| baseline | AG + transpose + RS (2 dispatches) | 8.63 | 86.62 | 816.5 ms | 2794.1 ms | 10.39 |

Agent is **1.245× faster** than the AG+RS baseline on the 1-node
5000-step training, **1.32× faster** on the 7-node 1000-step
steady-state, and **1.31× faster** per-call in 7-node training
(measured via `.item()` probe inside `_UA2A.forward/backward`).
The microbench-to-training ratio narrows from 2.35× to 1.31×
because adjacent `xm.all_gather` calls fuse inside the same
`mark_step` graph in real training. The agent extracts
each source rank's destined chunk via a
`for src in range(ws): chunks.append(gathered[src, rank*c:(rank+1)*c])`
loop followed by `torch.cat`. The cost model charges contiguity-aware
copies for the alternative
`narrow(non-leading-dim) → reshape → contiguous()` chain (which looks
metadata-only at the Python op level but moves the entire gathered
buffer at strided HBM bandwidth on every call), so the agent avoids
it. See *Cost-model design* below.

### Ring Attention KV

| Backend | Composition | Per-call HW (1-node ms) | Per-call training (7-node ms) | 5000-step avg (1-node) | 1000-step steady (7-node) | Final Loss |
|---|---|---:|---:|---:|---:|---:|
| **agent** | per-slot `all_gather` (2 dispatches: 1 K, 1 V) | **0.87** | **24.12** | **466.9 ms** | **1412.3 ms** | 10.32 |
| baseline (per-head naive) | per-head `all_gather` (16 K + 16 V = 32 dispatches/layer) | 4.73 | 25.99 | 474.0 ms | 1450.1 ms | 10.33 |

Agent is **1.015× faster** than the per-head naive baseline at
5000-step 1-node training scale, **1.026× faster** on
7-node 1000-step steady-state, and **1.08× faster** per-call
in 7-node training (`.item()` probe inside `_KVGather.forward`).
The microbench-to-training ratio collapses from 5.43× to 1.08×. The per-call HW gap is much wider (5.4×):
in isolated 20-iter microbench the per-head naive launches 32
collectives back-to-back without graph-level fusion, while the
per-slot variant has only 2. In real training the Neuron compiler
fuses adjacent `xm.all_gather` calls inside one `mark_step` graph into
a small number of NEFFs, so most of the per-call gap disappears
amortized over the layer's other compute. The 7 ms/step gap (≈80 s
over a 5000-step run) is small but consistent, and the agent's design
avoids dependence on the compiler's fusion behavior — useful insurance
across vendor library versions.

## 7-Node OLMoE-Style End-to-End Comparison

A single MoE training loop that exercises **two** of the searched
collectives simultaneously — AllToAllV and distributed cross-entropy
— inside `training/train_olmoe10b.py`, run on **7× trn1.32xlarge
(224 ranks)**. (Replicated-gradient AllReduce was originally also
searched; after a follow-up 19-smoke investigation showed no
non-obvious composition beats the per-tensor loop on this Trainium
stack — where AR latency is approximately payload-independent — we
keep it at the baseline per-tensor loop in both stacks.)

Architecture: **OLMoE-architectural-style** (no MLA, no shared experts,
standard multi-head attention + RoPE + RMSNorm + SwiGLU MoE) with
**expert-choice routing** so per-(src,dst) send/receive counts are
fixed at `cap` tokens each. With `NEXP = ws = 224` (one expert per rank)
the AllToAllV call becomes a uniform-shape exchange with no per-step
host-side count computation.

| | |
|---|---|
| LAYERS | 8 |
| DM | 2048 |
| HEADS | 16 |
| NEXP | 224 (1 expert per rank, expert-choice routing) |
| TOPK | 8 (average expert calls per token) |
| EXDIM | 1024 |
| SEQLEN | 256, BSZ=1, bf16 |
| VOCAB | 32256 (= 224 · 144, sharded across ranks for dxe) |
| cap | `ceil(SEQLEN · TOPK / NEXP) · 1.0 = 10` tokens per (src expert, dst rank) |
| Total params | ~11.3 B at world=224 (replicated 197 M, expert 50 M × 224) |

Two independently pluggable collective backends — each can be
`baseline` (developer-written reference) or `agent` (the 7-node search
output in `runtime/trainium_*_7node.py`):

| Collective | Baseline | Agent |
|---|---|---|
| AllToAllV (MoE dispatch + combine) | AG + reshape + RS-SUM | pack + 1 `all_gather` + `index_select` (`runtime/trainium_alltoallv_7node.py`) |
| Distributed cross-entropy | `all_gather` full logits + `F.cross_entropy` | 2 `all_reduce`s, no max-shift, bf16-safe (`runtime/trainium_dxe_7node.py`) |
| Replicated-gradient AllReduce | per-tensor `xm.all_reduce` loop (kept in both stacks; agent variant in `runtime/trainium_grad_ar_7node.py` lost in real training — see below) | per-tensor `xm.all_reduce` loop (baseline) |

<!-- OLMOE_RESULTS_BEGIN -->
### OLMoE 7-node baseline vs agent comparison (full stack)

World size: 224 ranks (7× trn1.32xlarge). Steps: 1000 / 1000.

| Backend stack | Wall clock | Avg step | Steady-state step (≥200) | Final loss | Finite/Total |
|---|---:|---:|---:|---:|---:|
| baseline a2av + baseline dxe | 4456.7 s (74.3 min) | 4456.1 ms | 4337.6 ms | 10.5512 | 1000/1000 |
| **agent a2av + agent dxe**   | 3284.5 s (54.7 min) | 3283.8 ms | **3072.5 ms** | 10.5475 | 1000/1000 |

**Agent vs baseline speedups**
- Wall-clock: **1.357×**
- Avg step:   **1.357×**
- Steady-state step (after warmup): **1.412×**

Two collectives are simultaneously under test: the agent stack
swaps `runtime/trainium_alltoallv_7node.py` into the MoE block and
uses `runtime/trainium_dxe_7node.py` for the cross-entropy
computation over vocab-sharded logits. The replicated-gradient
AllReduce uses the per-tensor `xm.all_reduce` loop in both stacks
(see Section ## below on the dropped grad_ar problem).
<!-- OLMOE_RESULTS_END -->

How this differs from the per-problem microbenchmark above:
- AllToAllV is called in the MoE block twice per layer (dispatch +
  combine) under real NEFF cache pressure from the surrounding model
  graph; the gradient AllReduce and the distributed CE each run once
  per step. Per-step time reflects interaction effects (compile-graph
  diversity, fabric contention, sharded-head logits all-gather avoided
  by dxe) that the isolated microbenchmark doesn't surface.
- Loss is still random-data — this remains a wall-clock measurement,
  not a convergence study. Loss floors at `log(VOCAB) ≈ log(32256) ≈
  10.38`.

## Runtime Modules

```python
# AllToAllV (variable-length MoE dispatch)
from runtime.trainium_alltoallv import all_to_allv, init_alltoallv
init_alltoallv()
output = all_to_allv(x, send_counts)              # variable counts
# or, equivalent uniform shorthand:
from runtime.trainium_alltoallv import alltoallv
output = alltoallv(x, world_size, max_chunk)

# Uniform AllToAll (fixed-capacity MoE token exchange)
from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
init_uniform_a2a()
recv = uniform_a2a(send_buf, chunk_size)

# Ring Attention KV Distribution
from runtime.trainium_ring_kv import ring_kv_gather, init_ring_kv
init_ring_kv()
all_kv = ring_kv_gather([k_local, v_local])       # list of slot tensors -> list

# Replicated-gradient AllReduce (call after loss.backward(), before opt.step())
from runtime.trainium_grad_ar_7node import grad_ar_sync, init_grad_ar
init_grad_ar()
grad_ar_sync(rep_params, world_size)              # in-place on .grad

# Distributed cross-entropy on vocab-sharded logits
from runtime.trainium_dxe_7node import dxe_loss, init_dxe
init_dxe()
loss = dxe_loss(logits_local, targets, V_local)   # logits_local: (N, V_local)
```

The original AG+RS baseline is available separately:

```python
from runtime.ag_reduce_scatter import alltoallv, init_alltoallv
```

## How the Search Loop Works

```
Phase 1: Hardware Profiling — agent calls profiling tools
         (measure_collective_latency, measure_xla_op_overhead,
          measure_compilation_cost, measure_graph_launch_overhead,
          measure_memory_copy_throughput, ...)
         and writes its own cost model.

Phase 2: Baseline Evaluation — runs builtin templates through the
         simulator, builds a knowledgebase of which patterns work.

Phase 3: Multi-Island Evolution — 3+ independent islands, each starting
         from a different builtin template. The LLM synthesizes code →
         sandbox → multi-rank correctness test → simulator score. Per-op
         feedback per round.

Phase 4: Hardware Validation — top simulator candidates are benchmarked
         on real Trainium (`xm.rendezvous` + 20-iter timed loop), then
         passed through MoE training validation (10 steps, full autograd).
         Failed candidates get LLM-assisted recovery (≤2 attempts).
         The candidate with the best on-hardware latency wins.

Phase 5: Code Generation — winner written to runtime/trainium_<problem>.py.
```

The Phase 1 profiling tool surface is intentionally narrow but covers the
quirks that materially affect ranking on Trainium 1-node:

- `measure_collective_latency` — end-to-end latency of `all_gather`,
  `reduce_scatter`, `all_reduce`, `collective_permute`, `all_to_all` at
  varying tensor sizes/step counts.
- `measure_p2p_transfer` — point-to-point time between specific devices.
- `measure_xla_op_overhead` — per-op cost of every local XLA op.
- `measure_index_select_scaling` — `index_select` cost vs index/source size.
- `measure_compilation_cost(tensor_bytes)` — per-NEFF load cost as a
  function of largest single-collective tensor in the graph.
- `measure_graph_launch_overhead` — per-`mark_step` framework cost
  (host-device sync + tracing + dispatch) above per-collective microbench.
- `measure_memory_copy_throughput` — sequential vs strided on-device
  memcpy bandwidth. The strided regime is what an implicit copy from a
  non-contiguous source actually pays; without this the cost model
  cannot tell a metadata-only `view` chain from a chain whose `reshape()`
  silently invokes a real O(numel) copy.
- `check_cross_node_support` / `check_primitive_compilation` — hardware
  guardrails on which primitives compile and which patterns crash.

### Cost-Model Design

```
total_us = sum(per_op_cost(op, copy_bytes_if_any) for op in local_ops)
         + num_dispatches * dispatch_overhead
         + 2 * graph_launch_overhead         # one mark_step pair per autograd Function
         + compilation_cost(max_single_collective_tensor_bytes)
           * load_events_per_run / training_steps
         + bandwidth_term(volume / link_bw)
         + cross_node_penalty
```

Five design choices in the per-op cost function are worth calling out
because they materially change rankings at single-node scale and
prevent the LLM from reward-hacking the simulator:

1. **Graph-launch overhead is paid per `mark_step` pair, not per
   collective.** XLA fuses all collectives inside one `autograd.Function`
   into a single HLO graph, so an algorithm with N in-graph dispatches
   pays one launch cost, not N. Charging per-collective double-counted
   the cost for naive per-tensor loops.

2. **Each local XLA op has a `min_local_op_us` floor (default 1 µs).**
   Without this, an algorithm replacing real compute ops (`cat`, `slice`,
   `sum`) with metadata-only ops (`view`, `narrow`, `reshape`) would
   score arbitrarily close to zero.

3. **Pure metadata view ops are floor-priced.** `view`, `narrow`,
   `transpose`, `permute`, `expand`, `squeeze`, `unsqueeze`, `flatten`,
   `slice` are PyTorch view operations — they never copy. Their isolated
   `measure_xla_op_overhead` value (~28 µs) reflects mark_step-boundary
   kernel launch overhead, not their cost when fused alongside other
   ops in a single HLO graph. Charging the isolated value penalized
   slice-loop chains that XLA actually fuses.

4. **`reshape()` and `contiguous()` charge an implicit-copy term.**
   Both are *contiguity-dependent* — free when the input is already
   contiguous in the requested order, otherwise PyTorch silently
   inserts an O(numel) copy of the source storage. The simulator
   detects which case applies by inspecting actual tensor strides at
   trace time and charges
   `max(view_floor, scaled_bytes / strided_memcpy_bw)`. This is the
   change that prevented the agent from picking a Uniform AllToAll
   variant whose receive-side extraction was
   `narrow(dim=1, rank, 1) → reshape(-1) → contiguous()` — a four-op
   chain that scored ~0.4 µs in the old model but moves the entire
   gathered buffer at strided bandwidth in real training.

5. **`index_select` and `torch.tensor(python_list, ...)` are
   volume-scaled.** The agent's first ring_kv design was a
   `cat → 1 all_gather → torch.tensor(huge_python_index) → index_select`
   chain. Both `index_select` (which moves output bytes at random-access
   bandwidth) and `torch.tensor(python_list)` (which does an O(N) host-
   side construction plus a host→device copy) had flat ~29 µs costs in
   the old model regardless of size; at training scale the index list
   was 16 M elements and the per-step cost was 17 *seconds*. The
   simulator now charges
   `max(agent_floor, scaled_bytes / strided_memcpy_bw)` for both.

6. **Phase 5 winner ranks by simulator score among HW+TV survivors,
   not isolated-microbench latency.** The 20-iter `xla.step()` HW
   microbench measures isolated-call latency — fast first-iter NEFF
   compile, then 19 cached-NEFF iters. It does not surface NEFF cache
   eviction under `NEURON_NUM_RECENT_MODELS_TO_KEEP=1` or the
   per-mark_step framework overhead that dominates 5000-step training.
   Among candidates that pass HW microbench AND 10-step bf16 training
   validation (gates), Phase 5 picks the lowest *simulator* score.
   This matters even for the AllToAllV problem: per-call microbench
   prefers the AG+RS baseline (fewer local ops), but at 5000-step
   training scale the agent's pack-and-`index_select` composition wins
   because of amortized compilation cost.

7. **`cat` / `stack` are bytes-aware, and back-to-back collective
   dispatches amortize.** Two physics terms the count-based model
   missed:
   - `torch.cat` and `torch.stack` always allocate a new contiguous
     buffer and copy all inputs through HBM. The cost model charges
     `max(per_op_floor, bytes / sequential_memcpy_bw)` for them,
     keyed off the output bytes recorded at trace time. Without this,
     a 32 MB cat scored the same as a 32 KB cat — a free pass for
     concat-then-single-collective patterns at training scale.
   - A run of `xm.all_reduce` (or other collective) issues with no
     intervening data-consume between them pipelines into the EFA NIC:
     only the first issue in the run pays the full per-dispatch setup,
     subsequent issues pay an amortized cost (default ~10% of full
     setup). The simulator walks the recorded event stream in
     chronological order and charges per-issue accordingly. Run length
     resets on any non-`_FREE_XLA_OPS` op.

   Both terms are algorithm-agnostic: the simulator never inspects
   which algorithm it is scoring, only the recorded event stream.
   Their effect on the gradient-AR ranking was to lift
   `async_back_to_back` from ~11400 µs (count-based) to ~5800 µs
   (event-aware), within 10% of the 1-AR `flat_single_ar` family, which
   matches the 5% spread the HW microbench sees on the same shapes.

8. **TrackedTensor arithmetic dunders record into the op counter, and
   elementwise ops are floor-priced.** On XLA each `*`/`+`/`-`/`/`/`%`
   between TrackedTensors lowers to a distinct HLO node, so an
   algorithm that interleaves a `* inv` between every `xm.all_reduce`
   has a different event stream from one that issues all AllReduces
   first and applies `* inv` afterwards. Without the dunder
   `self._counter.record("mul")` call, those two patterns scored
   identically and the back-to-back-amortization in (7) had no signal
   to act on. Conversely, `mul`/`add`/`sub`/`div`/`mod`/`neg` are
   placed in `_FUSED_ELEMENTWISE_OPS` and priced at the per-op floor,
   reflecting the XLA HLO fact that adjacent elementwise ops fuse into
   a single kernel.

The simulator computes copy-bytes from the actual PyTorch tensor at
trace time (via `is_contiguous()` and shape compatibility checks); the
agent never sees the per-op cost numbers, so this isn't a "leak" of
which algorithm is best — it's a physics correction that any agent
must navigate. Algorithms that hide a real memory copy behind metadata
ops are correctly priced by the simulator without the prompt telling
the agent to avoid any specific pattern.

## Running the Agent

```bash
# Run search for any collective problem (1-node default)
python experiments/run_search.py --problem alltoallv --pattern moe
python experiments/run_search.py --problem uniform_a2a
python experiments/run_search.py --problem ring_kv
python experiments/run_search.py --problem grad_ar
python experiments/run_search.py --problem dxe

# With on-hardware validation (recommended — picks HW-fastest candidate)
python experiments/run_search.py --problem ring_kv --hw-eval --num-nodes 1

# Multi-node hardware evaluation
python experiments/run_search.py --problem alltoallv --hw-eval \
    --num-nodes 2 --master-addr 172.31.48.122 --worker-addrs 172.31.55.245

# Pick the LLM model (default: opus; sonnet recommended for cost)
python experiments/run_search.py --problem ring_kv --llm-model sonnet --max-rounds 8

# Search without LLM (heuristics + GA + SA only — fast iteration)
python experiments/run_search.py --pattern moe --no-llm
```

### Cost & Wall Time per Problem

End-to-end cost of running the full search (Sonnet 4.6,
`--max-rounds 8 --llm-candidates 3 --hw-eval`) on a single trn1.32xlarge:

| Problem | Wall (min) | LLM calls | Input tokens | Output tokens | Cost (USD) |
|---|---:|---:|---:|---:|---:|
| AllToAllV | 35.2 | 84 | 617,596 | 85,956 | $3.14 |
| Uniform AllToAll | 30.9 | 47 | 411,068 | 77,521 | $2.40 |
| Ring Attention KV | 21.9 | 39 | 388,359 | 53,058 | $1.96 |
| **Total** | **88.0** |  | **1,417,023** | **216,535** | **$7.50** |

Pricing assumes Sonnet 4.6 list rates ($3 / 1M input, $15 / 1M output).
Per-problem wall is dominated by Phase 3 LLM evolution (~75 % of LLM
calls) plus Phase 4 on-Trainium hardware microbenchmarks (~5–10 min per
problem, including Neuron NEFF compilation for unfamiliar candidate
graphs).

## Lessons / open issues

The cost model is now physics-aware enough that the committed
runtimes match or beat the developer-written baseline at 5000-step
training scale, but it still has known limits:

- **Strided vs sequential memcpy bandwidth uses two constants per
  device.** The simulator distinguishes dense-permute copies (full
  storage, predictable strides → sequential bw) from sub-region copies
  (narrow on non-leading dim → strided bw). Real Trainium HBM
  strided bandwidth depends on the actual stride pattern (transposed
  reads vs sub-cache-line gathers vs partial-row writes), but
  `measure_memory_copy_throughput` collapses each regime to one number.
  Algorithms whose copies fall between regimes may be mis-priced.
- **`measure_compilation_cost` samples top out at 100 MB.** Above that,
  the simulator log-linearly extrapolates from the last two samples,
  which captures the super-linear NEFF growth direction but not its
  exact shape past the threshold.
- **No async / overlap modeling.** The simulator scores algorithms as
  if everything is serial. Anything that wins via forward-backward
  pipelining or compute-collective overlap is invisible to the search.
- **The 10-step training validation harness uses an 8-layer DIM=1024
  synthetic model.** This is large enough to force NEFF cache eviction
  under `NEURON_NUM_RECENT_MODELS_TO_KEEP=1` and surface per-step
  framework overhead, but its absolute step time (~75 ms) does not
  rank candidates the same way 5000-step DeepSeek-MoE-Lite training
  does (~420 ms/step). Phase 5 ranks by simulator score among
  HW-microbench-and-TV survivors instead, treating the gates as
  feasibility checks and the simulator as the per-step cost predictor.

The right next step on (1) is to extend `measure_xla_op_overhead` to
report a *conditional* cost (cheap if input contiguous, copy-cost
otherwise, with a stride-aware bandwidth term parameterized by access
pattern), and on (3) to add an overlap term parameterized by
(num_dispatches, average_dispatch_size). Both require new on-hardware
profiling rather than ad-hoc tuning of existing constants.

## Running Training

### 1-Node (1× trn1.32xlarge, 32 ranks)

```bash
torchrun --nproc_per_node=32 training/train.py --backend evolved --steps 5000
torchrun --nproc_per_node=32 training/train.py --backend agrs    --steps 5000

torchrun --nproc_per_node=32 training/train_uniform_a2a.py --backend evolved --steps 5000
torchrun --nproc_per_node=32 training/train_ring_kv.py     --backend evolved --steps 5000
```

### 2-Node (2× trn1.32xlarge, 64 ranks)

```bash
torchrun --nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER:29500 training/train.py --backend evolved --steps 5000

torchrun --nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER:29500 training/train_uniform_a2a.py --backend evolved --steps 5000
torchrun --nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER:29500 training/train_ring_kv.py     --backend evolved --steps 5000
```

`MASTER` should be the master node's private IP. Both nodes must be in
the same VPC with EFA configured.

### 7-Node (7× trn1.32xlarge, 224 ranks)

The OLMoE-style end-to-end run lives behind one launcher,
`training/run_olmoe_compare.sh`. Each of the two collectives has an
independent `baseline|agent` toggle:

```bash
# Full developer-written baseline stack (1000 steps default).
bash training/run_olmoe_compare.sh baseline

# Full agent stack: AllToAllV + dxe from runtime/*_7node.py (grad_ar uses baseline per-tensor loop)
GRAD_SYNC=agent CE=agent bash training/run_olmoe_compare.sh agent
```

`MASTER` and `WORKERS` inside the launcher are the cluster's private
IPs; the launcher rsyncs the repo to all workers and runs
`torchrun --nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d` per node.
JSON results land in `training/results/olmoe_7node/`.

## Topology

16 NeuronDevices in a 2D 4×4 torus per node, 2 NeuronCores each (32 ranks/node).
4 bidirectional NeuronLinks per device (~192 GB/s). Inter-node: 8 EFA
adapters @ 12.5 GB/s each.

```
Device adjacency (from neuron-ls):
  0:[12,3,4,1]    4:[0,7,8,5]    8:[4,11,12,9]   12:[8,15,0,13]
  1:[13,0,5,2]    5:[1,4,9,6]    9:[5,8,13,10]   13:[9,12,1,14]
  2:[14,1,6,3]    6:[2,5,10,7]  10:[6,9,14,11]   14:[10,13,2,15]
  3:[15,2,7,0]    7:[3,6,11,4]  11:[7,10,15,8]   15:[11,14,3,12]
```

## Correctness Verification

Every accepted composition passes 5 layers:

1. **Reference implementation** — pure-Python ground truth per problem.
2. **Multi-rank simulation** — `TrackedTensor` + `CollectiveSimulator`
   across world sizes (4, 8) and multiple traffic patterns.
3. **bf16 correctness** — candidates tested with bfloat16 inputs.
4. **Sandbox execution** — restricted environment, no file I/O or imports.
5. **Real hardware verification** — diagnostic inputs with post-execution
   checks; full MoE 10-step training run.

## Requirements

- AWS Trainium instance (trn1.32xlarge) with `neuronxcc` and `torch-neuronx`
- `boto3` for Bedrock (default), or `ANTHROPIC_API_KEY` for direct API
- Multi-node: 2+ trn1.32xlarge in the same VPC with SSH + EFA
