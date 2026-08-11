def evolved_p99(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i*i if i < N/2 else (N-i)*(N-i)
    idx = torch.arange(N, device=x.device)
    m = torch.where(idx < N // 2, idx, N - idx)
    return (m * m).to(x.dtype)

