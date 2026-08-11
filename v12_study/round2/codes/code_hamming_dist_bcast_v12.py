def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    vals = [[bin(i ^ j).count("1") for j in range(N)] for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

