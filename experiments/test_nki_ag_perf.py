"""Profile NKI nccl.all_gather vs XLA xm.all_gather overhead."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl_mod
import torch
import torch.distributed as dist
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import time

WORLD = 32


@nki.jit
def nki_ag_only(input_hbm):
    """Minimal NKI all_gather: no pack, no extract."""
    gathered = nl.ndarray((1, WORLD * 4096), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    nccl_mod.all_gather(op=np.add, srcs=[input_hbm], dsts=[gathered],
                        replica_groups=[list(range(WORLD))],
                        all_gather_dim=1, dtype=nl.float32)
    return gathered


def main():
    device = xla.device()
    dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    world = xr.world_size()

    SIZE = 4096
    x = torch.randn(1, SIZE, device=device, dtype=torch.float32)
    warmup = 10
    iters = 50

    # --- NKI all_gather ---
    for _ in range(warmup):
        y = nki_ag_only(x)
        xla.step()
    xm.wait_device_ops()

    start = time.perf_counter()
    for _ in range(iters):
        y = nki_ag_only(x)
        xla.step()
    xm.wait_device_ops()
    nki_ms = (time.perf_counter() - start) / iters * 1000

    # --- XLA all_gather ---
    for _ in range(warmup):
        g = xm.all_gather(x, dim=0)
        _ = g.sum()  # anti-DCE
        xla.step()
    xm.wait_device_ops()

    start = time.perf_counter()
    for _ in range(iters):
        g = xm.all_gather(x, dim=0)
        _ = g.sum()
        xla.step()
    xm.wait_device_ops()
    xla_ms = (time.perf_counter() - start) / iters * 1000

    # --- XLA all_gather with full tensor accumulation (anti-DCE) ---
    accum = torch.zeros(world, SIZE, device=device, dtype=torch.float32)
    for _ in range(warmup):
        g = xm.all_gather(x, dim=0)
        accum += g
        xla.step()
    xm.wait_device_ops()

    start = time.perf_counter()
    for _ in range(iters):
        g = xm.all_gather(x, dim=0)
        accum += g
        xla.step()
    xm.wait_device_ops()
    xla_full_ms = (time.perf_counter() - start) / iters * 1000

    if rank == 0:
        print(f"NKI nccl.all_gather:     {nki_ms:.4f} ms ({SIZE} elems/rank)")
        print(f"XLA xm.all_gather+sum:   {xla_ms:.4f} ms ({SIZE} elems/rank)")
        print(f"XLA xm.all_gather+accum: {xla_full_ms:.4f} ms ({SIZE} elems/rank)")
        print(f"NKI/XLA ratio:           {nki_ms/xla_full_ms:.2f}x")


if __name__ == "__main__":
    main()
