"""Test NKI nccl.all_gather on real Trainium hardware.

Key findings from llama3_transformer.py:
- all_gather(op=np.add, srcs=[...], dsts=[...], replica_groups=..., all_gather_dim=0)
- replica_groups must be list-of-lists: [[0,1,...,31]]
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
import torch
import torch.distributed as dist
import torch_xla as xla
import torch_xla.runtime as xr

WORLD = 32


@nki.jit
def nki_ag_test(input_hbm):
    """All_gather: each rank has (1, 4) -> gather to (1, 4*WORLD)."""
    ELEM = 4
    replica_groups = [list(range(WORLD))]

    local_buf = nl.ndarray((1, ELEM), dtype=nl.float32, buffer=nl.shared_hbm)
    d = nl.load(input_hbm[0:1, 0:ELEM])
    nl.store(local_buf[0:1, 0:ELEM], d)

    gathered = nl.ndarray((1, WORLD * ELEM), dtype=nl.float32,
                          buffer=nl.shared_hbm)
    nccl.all_gather(op=np.add, srcs=[local_buf], dsts=[gathered],
                    replica_groups=replica_groups,
                    all_gather_dim=1, dtype=nl.float32)

    return gathered


def main():
    device = xla.device()
    dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    world = xr.world_size()
    assert world == WORLD

    # Each rank has 4 elements: [rank*10, rank*10+1, rank*10+2, rank*10+3]
    ELEM = 4
    x = torch.zeros(1, ELEM, device=device, dtype=torch.float32)
    for i in range(ELEM):
        x[0, i] = float(rank * 10 + i)

    gathered = nki_ag_test(x)
    xla.step()

    g_cpu = gathered.cpu()
    ok = True
    for src in range(world):
        for i in range(ELEM):
            expected = float(src * 10 + i)
            actual = g_cpu[0, src * ELEM + i].item()
            if abs(expected - actual) > 1e-3:
                ok = False
                if rank == 0:
                    print(f"Mismatch at src={src}, i={i}: "
                          f"expected={expected}, got={actual}")

    if rank == 0:
        print(f"NKI nccl.all_gather: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
