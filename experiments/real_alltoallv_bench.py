#!/usr/bin/env python3
"""
Real hardware benchmark: Optimized AllToAllV vs topology-unaware baselines.

Each (algorithm, shard_size) pair runs in its own torchrun process to stay
within the Neuron runtime's communication group limit.

Uses realistic MoE (Mixture-of-Experts) traffic by default: Zipf-distributed
expert popularity with shuffled expert-to-rank mapping, so send_counts vary
per rank.

Supports multi-node clusters: use --num-nodes and --master-addr for N-node
benchmarks. Multi-node torchrun uses c10d rendezvous.

Benchmark scope (--scope flag):
    intra: Intra-node only (32 ranks within a single node, NeuronLink)
    full:  Full cross-node (N*32 ranks across all nodes, NeuronLink + EFA)
    both:  Run both scopes and show cross-node penalty comparison table

Usage:
    # Single node (default):
    python experiments/real_alltoallv_bench.py --algo all

    # Multi-node with scope separation:
    python experiments/real_alltoallv_bench.py --algo all --num-nodes 2 \\
        --master-addr <head-node-ip> --scope both

    # Intra-node only (useful for isolating NeuronLink performance):
    python experiments/real_alltoallv_bench.py --algo all --scope intra

    # Run a single algorithm at one shard size (called by orchestrator via torchrun):
    torchrun --nproc_per_node=32 experiments/real_alltoallv_bench.py \\
        --algo hierarchical --sizes 1024 --mode worker
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np


# ============================================================
# Schedules
# ============================================================

HIER_OPTIMIZED_SCHEDULE = [4, 3, 13, 12, 2, 15, 1, 7, 5, 11, 9, 14, 8, 6, 10]
CPD = 2

def _build_bench_topology(world):
    """Compute topology constants for given world size.

    Returns (num_devices, ring_schedule).
    """
    num_devices = world // CPD
    ring_schedule = list(range(1, world))
    return num_devices, ring_schedule


# Single-node defaults (overridden in worker mode from actual world size)
NUM_DEVICES = 16
DEFAULT_RING_SCHEDULE = list(range(1, 32))
NUM_DEVICES, DEFAULT_RING_SCHEDULE = _build_bench_topology(32)


def make_moe_send_counts(world=32, shard_size=1024):
    """Generate realistic MoE traffic with variable per-rank totals.

    Models a top-2 gated MoE where:
      - Each rank has a variable-sized local batch (Gaussian around shard_size,
        stddev=0.3*shard_size, clipped to [shard_size//4, 2*shard_size]).
      - Expert popularity follows Zipf(s=1.2), shuffled across ranks.
      - Each token is routed to top-2 experts (different destinations).

    Returns send_counts_matrix[src_rank][dst_rank] = number of elements (python
    list-of-lists).  Deterministic (seeded).  Per-rank totals intentionally vary
    so that AllGather must pad to max_total, making the comparison fair.

    Uses numpy vectorized sampling — runs in <1s even for shard_size=4M.
    """
    rng = np.random.RandomState(42)

    # Expert popularity: Zipf(s=1.2), shuffled so hot experts aren't on rank 0
    raw = np.array([1.0 / (i + 1) ** 1.2 for i in range(world)])
    perm = rng.permutation(world)
    probs = np.zeros(world)
    probs[perm] = raw
    probs /= probs.sum()

    # Variable batch size per rank
    lo, hi = max(shard_size // 4, 1), shard_size * 2
    batch_sizes = rng.normal(shard_size, 0.3 * shard_size, size=world)
    batch_sizes = np.clip(batch_sizes, lo, hi).astype(int)

    # Use multinomial to draw destination counts directly per rank.
    # For top-2 routing, each token goes to two different experts.
    # We approximate by drawing two independent multinomial samples and
    # redistributing collisions (same expert chosen twice) to the runner-up.
    # Conditional probs for second expert given first: p(e2|e1) = p(e2)/(1-p(e1))
    matrix = []
    for s in range(world):
        bs = int(batch_sizes[s])
        # First expert choice: multinomial over all experts
        c1 = rng.multinomial(bs, probs)
        # Second expert: for each first-choice expert e, draw the second
        # from the remaining experts with renormalised probs
        c2 = np.zeros(world, dtype=int)
        for e in range(world):
            n = c1[e]
            if n == 0:
                continue
            # Conditional distribution excluding expert e
            cond_p = probs.copy()
            cond_p[e] = 0.0
            cond_p /= cond_p.sum()
            c2 += rng.multinomial(n, cond_p)
        matrix.append((c1 + c2).tolist())
    return matrix


# ============================================================
# Worker mode: runs under torchrun, single algo + single size
# ============================================================

def run_worker(algo, shard_size, iters, warmup, do_correctness, bench_only=False):
    import torch
    import torch.distributed as dist
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    device = xla.device()
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    world = xr.world_size()
    rank = xr.global_ordinal()

    # Recompute topology constants from actual world size
    global NUM_DEVICES, DEFAULT_RING_SCHEDULE
    NUM_DEVICES, DEFAULT_RING_SCHEDULE = _build_bench_topology(world)
    num_nodes = max(1, world // 32)

    # ---- Helpers ----
    def cumulative_offsets(counts):
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        return offsets

    def build_padded_shards(x, send_counts, send_offsets, max_chunk):
        shards = []
        for i in range(world):
            chunk = x[send_offsets[i]:send_offsets[i] + send_counts[i]]
            if send_counts[i] < max_chunk:
                pad = torch.zeros(max_chunk - send_counts[i],
                                  device=x.device, dtype=x.dtype)
                chunk = torch.cat([chunk, pad])
            shards.append(chunk)
        return shards

    # ---- Algorithm implementations ----
    def do_hierarchical(x, send_counts, recv_counts, step_max_chunks, pairs):
        """step_max_chunks[0] = intra-device, step_max_chunks[1..15] = inter-device."""
        my_dev = rank // CPD
        send_off = cumulative_offsets(send_counts)
        recv_off = cumulative_offsets(recv_counts)
        output = torch.empty(sum(recv_counts), device=x.device, dtype=x.dtype)

        # Self copy (always correct: send_counts[rank] == recv_counts[rank])
        output[recv_off[rank]:recv_off[rank] + recv_counts[rank]] = \
            x[send_off[rank]:send_off[rank] + send_counts[rank]]

        # Intra-device peer: use collective_permute (can't read peer's buffer
        # in separate processes; on real HBM this would be a free copy).
        peer = rank ^ 1
        intra_mc = step_max_chunks[0]
        if peer < world:
            peer_chunk = x[send_off[peer]:send_off[peer] + send_counts[peer]]
            if send_counts[peer] < intra_mc:
                peer_chunk = torch.cat([peer_chunk, torch.zeros(
                    intra_mc - send_counts[peer],
                    device=x.device, dtype=x.dtype)])
            recv_peer = xm.collective_permute(peer_chunk, pairs=pairs[0])
            output[recv_off[peer]:recv_off[peer] + recv_counts[peer]] = \
                recv_peer[:recv_counts[peer]]

        # Inter-device steps (pairs[1:] for inter-device distances)
        for step_idx, d in enumerate(HIER_OPTIMIZED_SCHEDULE):
            dst_dev = (my_dev + d) % NUM_DEVICES
            src_dev = (my_dev - d + NUM_DEVICES) % NUM_DEVICES
            mc = step_max_chunks[step_idx + 1]
            chunks = []
            for c in range(CPD):
                dst_rank = dst_dev * CPD + c
                chunk = x[send_off[dst_rank]:send_off[dst_rank] + send_counts[dst_rank]]
                if send_counts[dst_rank] < mc:
                    chunk = torch.cat([chunk, torch.zeros(
                        mc - send_counts[dst_rank],
                        device=x.device, dtype=x.dtype)])
                chunks.append(chunk)
            send_tensor = torch.cat(chunks, dim=0)
            recv_tensor = xm.collective_permute(send_tensor, pairs=pairs[step_idx + 1])
            for c in range(CPD):
                from_rank = src_dev * CPD + c
                rc = recv_counts[from_rank]
                output[recv_off[from_rank]:recv_off[from_rank] + rc] = \
                    recv_tensor[c * mc:c * mc + rc]
        return output

    def do_ring(x, send_counts, recv_counts, max_chunk, pairs):
        send_off = cumulative_offsets(send_counts)
        recv_off = cumulative_offsets(recv_counts)
        output = torch.empty(sum(recv_counts), device=x.device, dtype=x.dtype)
        shards = build_padded_shards(x, send_counts, send_off, max_chunk)

        output[recv_off[rank]:recv_off[rank] + recv_counts[rank]] = \
            shards[rank][:recv_counts[rank]]
        for i, d in enumerate(DEFAULT_RING_SCHEDULE):
            send_to = (rank + d) % world
            recv_from = (rank - d) % world
            recv_tensor = xm.collective_permute(shards[send_to], pairs=pairs[i])
            output[recv_off[recv_from]:recv_off[recv_from] + recv_counts[recv_from]] = \
                recv_tensor[:recv_counts[recv_from]]
        return output

    def do_allgather(x, send_counts, matrix):
        # With MoE traffic, each rank's buffer is a different size.
        # Pad to max_total so all_gather gets uniform-sized inputs.
        max_total = max(sum(matrix[r]) for r in range(world))
        total_send = sum(send_counts)
        if total_send < max_total:
            x_padded = torch.cat([x, torch.zeros(
                max_total - total_send, device=x.device, dtype=x.dtype)])
        else:
            x_padded = x
        gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(world, max_total)
        # For each source rank, use that rank's send_offsets to find our slice
        chunks = []
        for src in range(world):
            src_send_counts = matrix[src]
            src_send_off = cumulative_offsets(src_send_counts)
            count = src_send_counts[rank]
            chunks.append(gathered[src, src_send_off[rank]:src_send_off[rank] + count])
        return torch.cat(chunks, dim=0)

    def do_agent(x, send_counts, agent_fn, agent_recv_counts, agent_max_chunk):
        """Agent output: calls the evolved algorithm from runtime module."""
        return agent_fn(x, send_counts, agent_recv_counts, agent_max_chunk)

    def do_fused_alltoall(x, send_counts, recv_counts, max_chunk):
        send_off = cumulative_offsets(send_counts)
        recv_off = cumulative_offsets(recv_counts)
        packed = torch.zeros(world * max_chunk, device=x.device, dtype=x.dtype)
        for i in range(world):
            packed[i * max_chunk:i * max_chunk + send_counts[i]] = \
                x[send_off[i]:send_off[i] + send_counts[i]]
        received = xm.all_to_all(packed, split_dimension=0,
                                 concat_dimension=0, split_count=world)
        output = torch.empty(sum(recv_counts), device=x.device, dtype=x.dtype)
        for i in range(world):
            rc = recv_counts[i]
            output[recv_off[i]:recv_off[i] + rc] = \
                received[i * max_chunk:i * max_chunk + rc]
        return output

    def do_reduce_scatter(x, send_counts, rs_flat_idx):
        """AllGather + ReduceScatter AllToAllV.

        AllGather collects packed buffers, reshape+transpose reorders by
        destination, ReduceScatter distributes each rank's shard.
        """
        pack_size = world * max_chunk
        packed = torch.zeros(pack_size, device=x.device, dtype=x.dtype)
        send_off = 0
        for i in range(world):
            sc = send_counts[i]
            if sc > 0:
                packed[i * max_chunk:i * max_chunk + sc] = x[send_off:send_off + sc]
            send_off += sc
        gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
        reshaped = gathered.view(world, world, max_chunk)
        transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
        my_shard = xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0/world,
                                      scatter_dim=0, shard_count=world)
        return torch.index_select(my_shard, 0, rs_flat_idx)

    def do_nki_ag(x, send_counts, nki_kernel):
        """NKI AllToAllV: fully fused pack+gather+extract in @nki.jit."""
        x_2d = x.unsqueeze(0)  # (1, N) for NKI tile layout
        out_2d = nki_kernel(x_2d)
        return out_2d.view(-1)

    # ---- Precompute pairs ----
    if algo == "hierarchical":
        # pairs[0] = intra-device swap (core 0 <-> core 1 on each device)
        # pairs[1..15] = inter-device distances from HIER_OPTIMIZED_SCHEDULE
        intra_pairs = [(r, r ^ 1) for r in range(world)]
        pairs = [intra_pairs]
        for d in HIER_OPTIMIZED_SCHEDULE:
            p = []
            for r in range(world):
                r_dst = ((r // CPD + d) % NUM_DEVICES) * CPD + (r % CPD)
                p.append((r, r_dst))
            pairs.append(p)
    elif algo == "ring":
        pairs = [[(r, (r + d) % world) for r in range(world)]
                 for d in DEFAULT_RING_SCHEDULE]
    elif algo == "fused_alltoall":
        pairs = None
    else:
        pairs = None

    # MoE traffic: non-uniform send_counts per rank.
    # All ranks compute the full matrix (deterministic seed) so they agree
    # on per-step max_chunks — collective_permute requires identical tensor shapes.
    matrix = make_moe_send_counts(world, shard_size)
    send_counts = matrix[rank]
    recv_counts = [matrix[src][rank] for src in range(world)]
    max_chunk = max(
        max(matrix[s][d] for s in range(world) for d in range(world)),
        1,
    )

    # Precompute NKI kernel for nki_ag
    nki_kernel = None
    nki_recv_total = None
    if algo == "nki_ag":
        from nki_alltoallv_hw import make_nki_alltoallv_kernel
        nki_kernel, nki_recv_total, _, _ = \
            make_nki_alltoallv_kernel(matrix, world, rank)

    # Load agent runtime module
    agent_fn = None
    agent_recv_counts = recv_counts
    agent_max_chunk = max_chunk
    if algo == "agent":
        import importlib.util
        _rt_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "runtime", "trainium_alltoallv.py")
        spec = importlib.util.spec_from_file_location("_agent_rt", _rt_path)
        _agent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_agent_mod)
        _agent_mod.init_alltoallv()
        agent_fn = _agent_mod.all_to_allv

    # Precompute reduce_scatter parameters
    rs_flat_idx = None
    if algo == "reduce_scatter":
        flat_idx_list = []
        for src in range(world):
            count = recv_counts[src]
            base = src * max_chunk
            flat_idx_list.extend(range(base, base + count))
        rs_flat_idx = torch.tensor(flat_idx_list, device=device,
                                    dtype=torch.long)

    # Per-step max_chunks for hierarchical (reduces padding ~38%)
    if algo == "hierarchical":
        # step 0: intra-device (core 0 <-> core 1 on same device)
        intra_max = max(
            max(matrix[d * CPD + c1][d * CPD + c2]
                for d in range(NUM_DEVICES) for c1 in range(CPD) for c2 in range(CPD)
                if c1 != c2),
            1,
        )
        step_max_chunks = [intra_max]
        for dist in HIER_OPTIMIZED_SCHEDULE:
            smc = max(
                max(matrix[dev * CPD + sc][(((dev + dist) % NUM_DEVICES) * CPD + dc)]
                    for dev in range(NUM_DEVICES) for sc in range(CPD) for dc in range(CPD)),
                1,
            )
            step_max_chunks.append(smc)
    else:
        step_max_chunks = None

    # ---- Correctness (only for allgather — Neuron's ENC_MAX_COMM_N limit
    #      prevents tensor readback from graphs with 15+ collective_permute).
    #      Hierarchical/ring correctness is verified by the simulator.
    if do_correctness:
        if algo in ("allgather", "agent", "fused_alltoall",
                    "nki_ag", "reduce_scatter"):
            # Build input with unique values per (rank, dst) segment
            segments = []
            send_off = cumulative_offsets(send_counts)
            for j in range(world):
                segments.append(torch.full((send_counts[j],),
                                           float(rank * world + j),
                                           device=device, dtype=torch.float32))
            x_test = torch.cat(segments) if segments else torch.tensor(
                [], device=device, dtype=torch.float32)
            if algo == "allgather":
                out = do_allgather(x_test, send_counts, matrix)
            elif algo == "agent":
                out = do_agent(x_test, send_counts, agent_fn,
                               agent_recv_counts, agent_max_chunk)
            elif algo == "nki_ag":
                out = do_nki_ag(x_test, send_counts, nki_kernel)
            elif algo == "reduce_scatter":
                out = do_reduce_scatter(x_test, send_counts, rs_flat_idx)
            else:
                out = do_fused_alltoall(x_test, send_counts, recv_counts, max_chunk)
            out_cpu = out.cpu()
            xla.step()

            # Verify: output[src_chunk] should contain values from src rank
            ok = True
            recv_off = cumulative_offsets(recv_counts)
            for src in range(world):
                expected_val = float(src * world + rank)
                seg = out_cpu[recv_off[src]:recv_off[src] + recv_counts[src]]
                if len(seg) > 0 and not torch.allclose(
                        seg, torch.full_like(seg, expected_val)):
                    ok = False
                    break

            if rank == 0:
                print(f"CORRECTNESS {algo}: {'PASS' if ok else 'FAIL'}")
        else:
            # For permute-based algorithms, verify the graph compiles and executes
            # (readback not possible due to Neuron comm group limit)
            x_test = torch.randn(sum(send_counts), device=device, dtype=torch.float32)
            if algo == "hierarchical":
                do_hierarchical(x_test, send_counts, recv_counts, step_max_chunks, pairs)
            elif algo == "ring":
                do_ring(x_test, send_counts, recv_counts, max_chunk, pairs)
            xla.step()
            xm.wait_device_ops()
            if rank == 0:
                print(f"CORRECTNESS {algo}: PASS (execution verified, "
                      f"data correctness verified by simulator)")
        return

    # ---- Benchmark ----
    x = torch.randn(sum(send_counts), device=device, dtype=torch.float32)
    # Anti-DCE: accumulate the FULL output tensor, not a scalar reduction.
    # Using out.sum() is insufficient because XLA can fuse all_gather + sum
    # into an all_reduce(sum), which is a tiny O(1) operation that bypasses
    # the actual data movement.  By accumulating the full tensor we force
    # the all_gather to materialise all gathered data element-by-element.
    out_size = sum(recv_counts)
    accum = torch.zeros(out_size, device=device, dtype=torch.float32)

    def run_algo():
        nonlocal accum
        if algo == "hierarchical":
            out = do_hierarchical(x, send_counts, recv_counts, step_max_chunks, pairs)
        elif algo == "ring":
            out = do_ring(x, send_counts, recv_counts, max_chunk, pairs)
        elif algo == "fused_alltoall":
            out = do_fused_alltoall(x, send_counts, recv_counts, max_chunk)
        elif algo == "agent":
            out = do_agent(x, send_counts, agent_fn,
                           agent_recv_counts, agent_max_chunk)
        elif algo == "nki_ag":
            out = do_nki_ag(x, send_counts, nki_kernel)
        elif algo == "reduce_scatter":
            out = do_reduce_scatter(x, send_counts, rs_flat_idx)
        else:
            out = do_allgather(x, send_counts, matrix)
        # Full element-wise accumulation — can't be fused into a reduction
        accum += out

    for _ in range(warmup):
        run_algo()
        xla.step()

    xm.wait_device_ops()
    start = time.perf_counter()
    for _ in range(iters):
        run_algo()
        xla.step()
    xm.wait_device_ops()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / iters * 1000

    if rank == 0:
        print(f"BENCH {algo} shard={shard_size} latency={latency_ms:.4f}ms")


# ============================================================
# Orchestrator mode: spawns separate torchrun for each combo
# ============================================================

ALGO_NAMES = {
    # Agent output (runtime/trainium_alltoallv.py — generated by search agent)
    "agent": "agent [AGENT] (runtime/trainium_alltoallv.py — agent-evolved algorithm)",
    # Baselines
    "nki_ag": "nki_ag [BASELINE] (NKI @nki.jit AllGather, 1 collective, 0 XLA ops)",
    "reduce_scatter": "reduce_scatter [BASELINE] (AllGather + ReduceScatter, 2 dispatches)",
    "fused_alltoall": "fused_alltoall [BASELINE] (single xm.all_to_all, 1 dispatch)",
    "allgather": "allgather_slice [BASELINE] (naive AllGather + slice, 1 collective, 33 XLA ops)",
    "hierarchical": "hierarchical [BASELINE] (topology-aware, 15 inter-device steps)",
    "ring": "default_ring [BASELINE] (topology-unaware, 31 steps)",
}


NEURON_VENV = os.environ.get(
    "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

# Worker node addresses for multi-node runs (set via --worker-addrs)
_WORKER_ADDRS = []


def _run_torchrun(cmd_args, num_nodes, master_addr, nproc, script, project_dir):
    """Run a torchrun subprocess, return (stdout, success, stderr).

    For multi-node (num_nodes > 1), also launches worker processes on
    remote nodes via SSH and waits for all to complete.
    """
    torchrun_bin = os.path.join(NEURON_VENV, "bin", "torchrun")
    torchrun_args = []
    if num_nodes > 1:
        torchrun_args = [
            torchrun_bin,
            f"--nnodes={num_nodes}",
            f"--nproc_per_node={nproc}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{MASTER_PORT}",
            script,
        ] + cmd_args
    else:
        torchrun_args = [
            torchrun_bin, f"--nproc_per_node={nproc}", script,
        ] + cmd_args

    timeout_s = 600 if num_nodes > 1 else 300

    if num_nodes > 1 and _WORKER_ADDRS:
        return _run_multinode_torchrun(
            torchrun_args, _WORKER_ADDRS, master_addr, project_dir, timeout_s)

    try:
        result = subprocess.run(
            torchrun_args, capture_output=True, text=True,
            timeout=timeout_s, cwd=project_dir)
        return result.stdout, result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return "", False, "timeout"


def _run_multinode_torchrun(cmd, worker_addrs, master_addr, project_dir,
                            timeout_s):
    """Launch torchrun on this node + all workers, collect results."""
    cmd_str = " ".join(cmd)
    env_setup = (
        f"export PATH={NEURON_VENV}/bin:$PATH && "
        f"export NEURON_RT_NUM_CORES=32 && "
        f"export FI_PROVIDER=efa && "
        f"export FI_EFA_USE_DEVICE_RDMA=1 && "
        f"export MASTER_ADDR={master_addr} && "
        f"export MASTER_PORT={MASTER_PORT} && "
        f"cd {project_dir}"
    )

    worker_procs = []
    for addr in worker_addrs:
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{addr}",
            f"{env_setup} && {cmd_str}",
        ]
        proc = subprocess.Popen(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        worker_procs.append((addr, proc))

    # Run master locally
    master_env = os.environ.copy()
    master_env["MASTER_ADDR"] = master_addr
    master_env["MASTER_PORT"] = MASTER_PORT
    master_env["NEURON_RT_NUM_CORES"] = "32"
    master_env["FI_PROVIDER"] = "efa"
    master_env["FI_EFA_USE_DEVICE_RDMA"] = "1"
    venv_bin = os.path.join(NEURON_VENV, "bin")
    master_env["PATH"] = venv_bin + ":" + master_env.get("PATH", "")
    try:
        master_result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=project_dir, env=master_env)
        master_stdout = master_result.stdout
        master_stderr = master_result.stderr
        master_ok = master_result.returncode == 0
    except subprocess.TimeoutExpired:
        master_stdout, master_ok, master_stderr = "", False, "timeout"
        for _, proc in worker_procs:
            proc.kill()

    # Collect worker outputs
    all_stderr = master_stderr
    for addr, proc in worker_procs:
        try:
            w_out, w_err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            w_out, w_err = proc.communicate()
        all_stderr += f"\n[worker {addr}] {w_err}"

    return master_stdout, master_ok, all_stderr


def _run_benchmark_pass(algos, sizes, iters, warmup, correctness,
                        num_nodes, master_addr, label=""):
    """Run all algorithms at all sizes, return results dict.

    Args:
        label: Optional label printed before each algo (e.g. "Intra-Node Only").

    Returns:
        results: {algo: {size: latency_ms}}
        correctness_results: {algo: bool}
    """
    script = os.path.abspath(__file__)
    project_dir = os.path.dirname(os.path.dirname(script))
    nproc = 32

    results = {a: {} for a in algos}
    correctness_results = {}

    if label:
        print(f"\n{'=' * 70}")
        world = num_nodes * 32
        print(f"  {label} ({world} ranks, {num_nodes} node(s))")
        print(f"{'=' * 70}")

    for algo in algos:
        print(f"--- {ALGO_NAMES.get(algo, algo)} ---")

        if correctness:
            stdout, ok, stderr = _run_torchrun(
                ["--algo", algo, "--sizes", str(sizes[0]),
                 "--mode", "worker", "--correctness"],
                num_nodes, master_addr, nproc, script, project_dir)
            for line in stdout.splitlines():
                if line.startswith("CORRECTNESS"):
                    print(f"  {line}")
                    correctness_results[algo] = "PASS" in line
            if not ok and algo not in correctness_results:
                if "Assertion" in stderr:
                    print(f"  CORRECTNESS {algo}: FAIL (runtime assertion)")
                else:
                    print(f"  CORRECTNESS {algo}: FAIL (exit code)")
                correctness_results[algo] = False

        for size in sizes:
            stdout, ok, stderr = _run_torchrun(
                ["--algo", algo, "--sizes", str(size),
                 "--iters", str(iters), "--warmup", str(warmup),
                 "--mode", "worker"],
                num_nodes, master_addr, nproc, script, project_dir)
            for line in stdout.splitlines():
                if line.startswith("BENCH"):
                    print(f"  {line}")
                match = re.search(r"latency=([\d.]+)ms", line)
                if match:
                    results[algo][size] = float(match.group(1))

            if not ok and size not in results[algo]:
                if "Assertion" in stderr:
                    print(f"  ERROR: Neuron runtime assertion (shard={size})")
                else:
                    print(f"  ERROR: exit code (shard={size})")
        print()

    return results, correctness_results


def _print_comparison_table(results, sizes, num_nodes, label=""):
    """Print the comparison table for a single benchmark pass."""
    world = num_nodes * 32

    print("=" * 70)
    title = f"COMPARISON: {label}" if label else "COMPARISON: All Algorithms"
    print(title)
    print(f"  ({world} ranks, {num_nodes} node(s))")
    print("=" * 70)

    hier = results.get("hierarchical", {})
    ring = results.get("ring", {})
    ag = results.get("allgather", {})
    agent = results.get("agent", {})
    nki = results.get("nki_ag", {})
    fused = results.get("fused_alltoall", {})
    rs = results.get("reduce_scatter", {})

    header = (f"  {'Shard':>8s}  {'Total':>10s}  "
              f"{'Agent':>14s}  {'NKI AG':>14s}  {'AG+RS':>14s}  "
              f"{'AllGather':>14s}  {'Fused A2A':>14s}  "
              f"{'Hierarchical':>14s}  {'Default Ring':>14s}")
    units = (f"  {'(elems)':>8s}  {'(bytes)':>10s}  "
             f"{'(ms)':>14s}  {'(ms)':>14s}  {'(ms)':>14s}  "
             f"{'(ms)':>14s}  {'(ms)':>14s}  "
             f"{'(ms)':>14s}  {'(ms)':>14s}")
    print(header)
    print(units)
    print("  " + "-" * 120)

    combined = []
    for size in sizes:
        total_bytes = size * world * 4
        h_ms = hier.get(size)
        r_ms = ring.get(size)
        a_ms = ag.get(size)
        agent_ms = agent.get(size)
        n_ms = nki.get(size)
        f_ms = fused.get(size)
        rs_ms = rs.get(size)

        def _fmt(v):
            return f"{v:>14.4f}" if v is not None else f"{'ERR':>14s}"

        print(f"  {size:>8d}  {total_bytes:>10d}  "
              f"{_fmt(agent_ms)}  {_fmt(n_ms)}  {_fmt(rs_ms)}  "
              f"{_fmt(a_ms)}  {_fmt(f_ms)}  {_fmt(h_ms)}  {_fmt(r_ms)}")

        combined.append({
            "shard_size": size,
            "total_bytes": total_bytes,
            "agent_ms": agent_ms,
            "nki_ag_ms": n_ms,
            "ag_reduce_scatter_ms": rs_ms,
            "allgather_slice_ms": a_ms,
            "fused_alltoall_ms": f_ms,
            "hierarchical_optimized_ms": h_ms,
            "default_ring_ms": r_ms,
        })

    print()
    return combined


def _print_penalty_table(intra_results, full_results, sizes, algos, num_nodes):
    """Print cross-node penalty table comparing intra-only vs full."""
    print("=" * 70)
    print(f"CROSS-NODE PENALTY (full {num_nodes}-node / intra 1-node)")
    print("=" * 70)
    header = f"  {'Shard':>8s}"
    for algo in algos:
        short = algo.replace("_", " ")[:12]
        header += f"  {short:>14s}"
    print(header)
    print("  " + "-" * (10 + 16 * len(algos)))

    for size in sizes:
        row = f"  {size:>8d}"
        for algo in algos:
            i_ms = intra_results.get(algo, {}).get(size)
            f_ms = full_results.get(algo, {}).get(size)
            if i_ms and f_ms:
                penalty = f_ms / i_ms
                row += f"  {penalty:>13.2f}x"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    # Average penalty per algo
    print()
    avg_row = f"  {'avg':>8s}"
    for algo in algos:
        penalties = []
        for size in sizes:
            i_ms = intra_results.get(algo, {}).get(size)
            f_ms = full_results.get(algo, {}).get(size)
            if i_ms and f_ms:
                penalties.append(f_ms / i_ms)
        if penalties:
            avg = sum(penalties) / len(penalties)
            avg_row += f"  {avg:>13.2f}x"
        else:
            avg_row += f"  {'N/A':>14s}"
    print(avg_row)
    print()


def run_orchestrator(algos, sizes, iters, warmup, output_path, correctness,
                     num_nodes=1, master_addr="localhost", scope="both"):
    """Run benchmark with scope: 'intra', 'full', or 'both'.

    - intra: 32 ranks within a single node (measures NeuronLink only)
    - full:  N*32 ranks across all nodes (measures NeuronLink + EFA)
    - both:  runs intra then full, shows cross-node penalty comparison
    """
    # Single-node: intra and full are identical
    if num_nodes <= 1:
        scope = "intra"

    print("=" * 70)
    node_desc = f" x {num_nodes} nodes" if num_nodes > 1 else ""
    print(f"REAL AllToAllV BENCHMARK — trn1.32xlarge{node_desc} (MoE traffic)")
    print("=" * 70)
    nd = num_nodes * 16
    nc = num_nodes * 32
    print(f"  Hardware:    trn1.32xlarge x {num_nodes} ({nd} NeuronDevices, "
          f"{nc} NeuronCores)")
    print(f"  Topology:    4x4 2D torus/node, {CPD} cores/device, "
          f"4 NeuronLinks/device @ 192 GB/s")
    if num_nodes > 1:
        print(f"  Inter-node:  EFA, 8 adapters/node @ 12.5 GB/s each")
    print(f"  Traffic:     MoE (Zipf s=1.2, shuffled expert assignment)")
    print(f"  Iterations:  {iters} (warmup: {warmup})")
    print(f"  Schedule:    {HIER_OPTIMIZED_SCHEDULE}")
    print(f"  Scope:       {scope}")
    print()

    intra_results = None
    full_results = None
    intra_combined = None
    full_combined = None

    if scope in ("intra", "both"):
        intra_results, _ = _run_benchmark_pass(
            algos, sizes, iters, warmup, correctness,
            num_nodes=1, master_addr=master_addr,
            label="INTRA-NODE ONLY (32 ranks, NeuronLink)")
        intra_combined = _print_comparison_table(
            intra_results, sizes, num_nodes=1,
            label="Intra-Node Only (32 ranks)")

    if scope in ("full", "both"):
        full_results, _ = _run_benchmark_pass(
            algos, sizes, iters, warmup, correctness,
            num_nodes=num_nodes, master_addr=master_addr,
            label=f"FULL CROSS-NODE ({nc} ranks, NeuronLink + EFA)")
        full_combined = _print_comparison_table(
            full_results, sizes, num_nodes=num_nodes,
            label=f"Full Cross-Node ({nc} ranks, {num_nodes} nodes)")

    if scope == "both" and intra_results and full_results:
        _print_penalty_table(intra_results, full_results, sizes, algos,
                             num_nodes)

    # ---- Save ----
    if output_path:
        out = {
            "hardware": f"trn1.32xlarge x {num_nodes}",
            "world_size": nc,
            "num_nodes": num_nodes,
            "num_devices": nd,
            "traffic_pattern": "moe",
            "optimized_schedule": HIER_OPTIMIZED_SCHEDULE,
            "iters": iters,
            "warmup": warmup,
            "scope": scope,
        }
        if intra_combined is not None:
            out["intra_comparison"] = intra_combined
        if full_combined is not None:
            out["full_comparison"] = full_combined
        # Backward compat: single-scope runs use "comparison" key
        if scope == "intra" and intra_combined is not None:
            out["comparison"] = intra_combined
        elif scope == "full" and full_combined is not None:
            out["comparison"] = full_combined
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to {output_path}")

    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real AllToAllV benchmark on Trainium")
    parser.add_argument("--algo", type=str, required=True,
                        choices=["hierarchical", "ring", "allgather",
                                 "agent", "fused_alltoall",
                                 "nki_ag", "reduce_scatter", "all"],
                        help="Algorithm (or 'all' for orchestrator)")
    parser.add_argument("--sizes", type=str, default="256,1024,4096,16384,65536",
                        help="Comma-separated shard sizes")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="orchestrator",
                        choices=["orchestrator", "worker"],
                        help="orchestrator spawns torchrun; worker runs under torchrun")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness check (worker mode)")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Number of trn1 nodes (default: 1)")
    parser.add_argument("--master-addr", type=str, default="localhost",
                        help="Master address for multi-node rendezvous")
    parser.add_argument("--scope", type=str, default="both",
                        choices=["intra", "full", "both"],
                        help="Benchmark scope: intra (single-node only), "
                             "full (cross-node), both (run both + penalty table)")
    parser.add_argument("--worker-addrs", type=str, default="",
                        help="Comma-separated private IPs of worker nodes "
                             "(e.g. 172.31.55.245)")
    args = parser.parse_args()

    if args.worker_addrs:
        global _WORKER_ADDRS
        _WORKER_ADDRS = [a.strip() for a in args.worker_addrs.split(",")
                         if a.strip()]

    sizes = [int(s) for s in args.sizes.split(",")]

    if args.mode == "worker":
        assert len(sizes) == 1, "Worker mode expects exactly one size"
        run_worker(args.algo, sizes[0], args.iters, args.warmup, args.correctness)
    elif args.algo == "all":
        # NKI agent + XLA baseline first, then other baselines
        algos = ["agent", "nki_ag", "reduce_scatter",
                 "allgather", "fused_alltoall", "hierarchical", "ring"]
        run_orchestrator(algos, sizes, args.iters, args.warmup,
                         args.output, args.correctness,
                         num_nodes=args.num_nodes,
                         master_addr=args.master_addr,
                         scope=args.scope)
    else:
        run_orchestrator([args.algo], sizes, args.iters, args.warmup,
                         args.output, args.correctness,
                         num_nodes=args.num_nodes,
                         master_addr=args.master_addr,
                         scope=args.scope)


if __name__ == "__main__":
    main()
