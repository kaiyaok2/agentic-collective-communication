def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    data = [[i ^ j for j in range(N)] for i in range(N)]
    return torch.tensor(data, device=x.device).to(x.dtype)
