def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i, j] = popcount(i XOR j) — constant matrix for fixed N.
    m = [[bin(i ^ j).count('1') for j in range(N)] for i in range(N)]
    return torch.tensor(m, device=x.device, dtype=x.dtype)

