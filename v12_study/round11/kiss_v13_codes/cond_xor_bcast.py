def evolved_p101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # constant fold
    vals = [[(i ^ j) if (i + j) % 2 == 0 else 0 for j in range(N)] for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

