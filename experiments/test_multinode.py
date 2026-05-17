#!/usr/bin/env python3
"""Quick inter-node connectivity test for Trainium multi-node AllToAllV.

Validates that:
  1. torch_xla initializes on both nodes
  2. All 64 ranks (2 nodes x 32 cores) can see each other
  3. xm.all_gather works across nodes
  4. xm.collective_permute works across nodes
  5. The evolved AllToAllV (all_gather + index_select) works across nodes

Usage (run as torchrun worker on each node):
    torchrun --nproc_per_node=32 --nnodes=2 \
        --rdzv_backend=c10d --rdzv_endpoint=<master>:29500 \
        experiments/test_multinode.py
"""

import time
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist


def main():
    device = xla.device()
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")

    world = xr.world_size()
    rank = xr.global_ordinal()
    node = rank // 32

    if rank == 0:
        print(f"[TEST 1] Initialization: {world} ranks across {world // 32} nodes")
        assert world == 64, f"Expected 64 ranks (2 nodes), got {world}"
        print(f"  PASS: {world} ranks initialized")

    # Test 2: all_gather across nodes
    x = torch.tensor([float(rank)], device=device)
    gathered = xm.all_gather(x)
    result = gathered.cpu()
    xla.step()

    if rank == 0:
        expected = torch.arange(world, dtype=torch.float32)
        ok = torch.allclose(result, expected)
        print(f"[TEST 2] Cross-node all_gather: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected: {expected[:5]}...{expected[-5:]}")
            print(f"  Got:      {result[:5]}...{result[-5:]}")

    # Test 3: collective_permute across nodes (shift by 32 = cross-node)
    x = torch.tensor([float(rank)], device=device)
    pairs = [(r, (r + 32) % world) for r in range(world)]
    received = xm.collective_permute(x, pairs=pairs)
    result = received.cpu()
    xla.step()

    if rank == 0:
        expected_val = float((rank - 32) % world)
        ok = abs(result.item() - expected_val) < 1e-5
        print(f"[TEST 3] Cross-node collective_permute (shift=32): "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Expected value at rank 0: {expected_val}, got {result.item()}")

    # Test 4: AllToAllV (evolved all_gather + index_select) across 64 ranks
    import numpy as np
    rng = np.random.RandomState(42)
    raw = np.array([1.0 / (i + 1) ** 1.2 for i in range(world)])
    perm = rng.permutation(world)
    probs = np.zeros(world)
    probs[perm] = raw
    probs /= probs.sum()

    # Build deterministic send_counts matrix (all ranks compute the same)
    matrix = []
    py_rng = __import__('random').Random(42)
    for s in range(world):
        counts = [0] * world
        for _ in range(256):
            r = py_rng.random()
            acc = 0.0
            for d in range(world):
                acc += probs[d]
                if r <= acc:
                    counts[d] += 1
                    break
        matrix.append(counts)

    send_counts = matrix[rank]
    recv_counts = [matrix[src][rank] for src in range(world)]

    # Build input with diagnostic values: float(src * world + dst)
    segments = []
    for dst in range(world):
        segments.append(torch.full((send_counts[dst],),
                                   float(rank * world + dst),
                                   device=device, dtype=torch.float32))
    x = torch.cat(segments) if segments else torch.tensor(
        [], device=device, dtype=torch.float32)

    # Evolved AllToAllV: all_gather + index_select
    max_total = max(sum(matrix[r]) for r in range(world))
    total_send = sum(send_counts)
    if total_send < max_total:
        x_padded = torch.cat([x, torch.zeros(
            max_total - total_send, device=x.device, dtype=x.dtype)])
    else:
        x_padded = x

    gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(-1)

    flat_idx = []
    for src in range(world):
        src_sc = matrix[src]
        src_off = 0
        for i in range(rank):
            src_off += src_sc[i]
        count = src_sc[rank]
        base = src * max_total + src_off
        flat_idx.extend(range(base, base + count))

    idx_tensor = torch.tensor(flat_idx, device=device, dtype=torch.long)
    output = torch.index_select(gathered, 0, idx_tensor)
    out_cpu = output.cpu()
    xla.step()

    # Verify
    if rank == 0:
        recv_off = [0]
        for c in recv_counts[:-1]:
            recv_off.append(recv_off[-1] + c)

        all_ok = True
        for src in range(world):
            expected_val = float(src * world + rank)
            seg = out_cpu[recv_off[src]:recv_off[src] + recv_counts[src]]
            if len(seg) > 0 and not torch.allclose(
                    seg, torch.full_like(seg, expected_val)):
                all_ok = False
                print(f"  FAIL at src={src}: expected {expected_val}, "
                      f"got {seg[:3]}")
                break

        cross_node_ok = False
        for src in range(32, world):
            if recv_counts[src] > 0:
                expected_val = float(src * world + rank)
                seg = out_cpu[recv_off[src]:recv_off[src] + recv_counts[src]]
                if torch.allclose(seg, torch.full_like(seg, expected_val)):
                    cross_node_ok = True
                    break

        print(f"[TEST 4] Cross-node AllToAllV (all_gather+index_select, "
              f"{world} ranks): {'PASS' if all_ok else 'FAIL'}")
        if cross_node_ok:
            print(f"  Verified: data from node 1 (ranks 32-63) received correctly")

    # Test 5: Latency benchmark (3 iterations)
    x_bench = torch.randn(total_send, device=device, dtype=torch.float32)
    if total_send < max_total:
        x_bench = torch.cat([x_bench, torch.zeros(
            max_total - total_send, device=x_bench.device, dtype=x_bench.dtype)])

    # Warmup
    for _ in range(2):
        g = xm.all_gather(x_bench.unsqueeze(0), dim=0).view(-1)
        _ = torch.index_select(g, 0, idx_tensor)
        xla.step()

    xm.wait_device_ops()
    start = time.perf_counter()
    for _ in range(5):
        g = xm.all_gather(x_bench.unsqueeze(0), dim=0).view(-1)
        _ = torch.index_select(g, 0, idx_tensor)
        xla.step()
    xm.wait_device_ops()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / 5 * 1000

    if rank == 0:
        print(f"[TEST 5] Cross-node AllToAllV latency: {latency_ms:.3f} ms "
              f"({world} ranks, 2 nodes)")
        print()
        print("All tests completed.")


if __name__ == "__main__":
    main()
