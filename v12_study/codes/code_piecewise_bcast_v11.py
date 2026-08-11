def evolved_p99(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i] = i*i if i < N/2 else (N-i)*(N-i)  ==  min(i, N-i)^2
    idx = torch.arange(N, device=x.device)
    m = torch.where(idx < N // 2, idx, N - idx)
    return (m * m).to(x.dtype)

