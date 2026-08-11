def evolved_p98(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i * 3 + 1) % (i % 7 + 2)
    vals = [(i * 3 + 1) % (i % 7 + 2) for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

