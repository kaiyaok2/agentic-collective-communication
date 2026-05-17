"""Test NKI collective_permute_implicit on real Trainium hardware."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.nccl.collectives import collective_permute_implicit
import torch
import torch.distributed as dist
import torch_xla as xla
import torch_xla.runtime as xr
import sys

WORLD = 32


@nki.jit
def nki_permute_test(input_hbm):
    """Send rank ID to next in ring, return what we receive."""
    ring = [list(range(WORLD))]

    send_buf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    recv_buf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    d = nl.load(input_hbm[0:1, 0:1])
    nl.store(send_buf[0:1, 0:1], d)

    collective_permute_implicit(
        src=send_buf[0:1, 0:1],
        dst=recv_buf[0:1, 0:1],
        replica_groups=ring,
        channel_id=0,
        num_channels=1,
    )

    output = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    d = nl.load(recv_buf[0:1, 0:1])
    nl.store(output[0:1, 0:1], d)
    return output


def main():
    device = xla.device()
    dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    world = xr.world_size()
    assert world == WORLD

    # Each rank sends its rank ID
    x_pad = torch.zeros(128, 1, device=device, dtype=torch.float32)
    x_pad[0, 0] = float(rank)

    y = nki_permute_test(x_pad)
    xla.step()

    received_from = int(y.cpu()[0, 0].item() + 0.5)
    # Print mapping for all ranks
    import torch_xla.core.xla_model as xm
    all_received = xm.all_gather(
        torch.tensor([received_from], device=device, dtype=torch.int32))
    xla.step()
    all_received_cpu = all_received.cpu().tolist()

    if rank == 0:
        print(f"Ring = [0..{WORLD-1}] (sequential)")
        print(f"collective_permute_implicit routing:")
        for r in range(world):
            print(f"  rank {r} received from rank {all_received_cpu[r]}")


if __name__ == "__main__":
    main()
