def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i,j] = popcount(i) + popcount(j); position-based, no collective
    pc = [bin(i).count("1") for i in range(N)]
    return torch.tensor([[a + b for b in pc] for a in pc],
                        device=x.device, dtype=x.dtype)

