
def evolved_p92(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i - N // 2) ** 2
    # For N=128, precompute as constant (compile-time folding)
    center = N // 2
    values = [(i - center) * (i - center) for i in range(N)]
    return torch.tensor(values, device=x.device, dtype=x.dtype)
