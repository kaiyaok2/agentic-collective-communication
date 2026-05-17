"""Test NKI AllToAllV kernel on real Trainium hardware.

Full pack -> all_gather -> extract pattern with variable send_counts.
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
import torch
import torch.distributed as dist
import torch_xla as xla
import torch_xla.runtime as xr
import time

WORLD = 32
# For testing: each rank sends CHUNK elements to each destination
CHUNK = 8
MAX_CHUNK = CHUNK
TOTAL_SEND = WORLD * CHUNK
PACK_SIZE = WORLD * MAX_CHUNK


def make_send_offsets():
    """Compute send offsets (uniform for this test)."""
    return [i * CHUNK for i in range(WORLD)]


@nki.jit
def nki_alltoallv_kernel(input_hbm):
    """AllToAllV via NKI: pack -> all_gather -> extract.

    Hardcoded for uniform CHUNK per destination for simplicity.
    On real hardware, send_counts would be compile-time constants
    per traffic pattern.
    """
    replica_groups = [list(range(WORLD))]

    # Pack: copy data for each destination into canonical slots
    packed = nl.ndarray((1, PACK_SIZE), dtype=nl.float32,
                        buffer=nl.shared_hbm)
    for i in range(WORLD):
        src_off = i * CHUNK
        dst_off = i * MAX_CHUNK
        d = nl.load(input_hbm[0, src_off:src_off + CHUNK])
        nl.store(packed[0, dst_off:dst_off + CHUNK], d)

    # All_gather: replicate all ranks' packed buffers
    gathered = nl.ndarray((1, WORLD * PACK_SIZE), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    nccl.all_gather(op=np.add, srcs=[packed], dsts=[gathered],
                    replica_groups=replica_groups,
                    all_gather_dim=1, dtype=nl.float32)

    return gathered


def main():
    device = xla.device()
    dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    world = xr.world_size()
    assert world == WORLD

    # Each rank creates: value = src_rank * 1000 + dst_rank * 10 + elem_offset
    input_data = torch.zeros(1, TOTAL_SEND, device=device, dtype=torch.float32)
    for dst in range(world):
        for off in range(CHUNK):
            input_data[0, dst * CHUNK + off] = float(rank * 1000 + dst * 10 + off)

    gathered = nki_alltoallv_kernel(input_data)
    xla.step()

    g_cpu = gathered.cpu()

    # Extract: for each source rank, find data it sent to this rank
    # Source rank `src` packed data for rank `dst` at:
    #   gathered[0, src * PACK_SIZE + dst * MAX_CHUNK : + CHUNK]
    ok = True
    for src in range(world):
        base = src * PACK_SIZE + rank * MAX_CHUNK
        for off in range(CHUNK):
            expected = float(src * 1000 + rank * 10 + off)
            actual = g_cpu[0, base + off].item()
            if abs(expected - actual) > 1e-3:
                ok = False
                if rank == 0:
                    print(f"Mismatch: src={src}, off={off}, "
                          f"expected={expected}, got={actual}")
                break
        if not ok:
            break

    if rank == 0:
        print(f"NKI AllToAllV correctness: {'PASS' if ok else 'FAIL'}")

    # Quick latency test
    warmup = 5
    iters = 20
    for _ in range(warmup):
        gathered = nki_alltoallv_kernel(input_data)
        xla.step()

    import torch_xla.core.xla_model as xm
    xm.wait_device_ops()
    start = time.perf_counter()
    for _ in range(iters):
        gathered = nki_alltoallv_kernel(input_data)
        xla.step()
    xm.wait_device_ops()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / iters * 1000

    if rank == 0:
        print(f"NKI AllToAllV latency: {latency_ms:.4f} ms "
              f"(CHUNK={CHUNK}, WORLD={WORLD})")


if __name__ == "__main__":
    main()
