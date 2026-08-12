def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    row = [[1 if (i + j) % 2 == 0 else -1 for j in range(N)] for i in range(N)]
    return torch.tensor(row, device=x.device, dtype=x.dtype)

