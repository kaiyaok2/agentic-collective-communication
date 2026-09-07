
def evolved_p87(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i * i) % K
    # For K=7, the pattern repeats: (i*i) % 7
    # Use constant folding for small N or compute directly
    if N <= 128:
        # Constant fold for small N
        values = [(i * i) % K for i in range(N)]
        return torch.tensor(values, device=x.device, dtype=x.dtype)
    else:
        # Arithmetic for larger N
        idx = torch.arange(N, device=x.device)
        result = (idx * idx) % K
        return result.to(x.dtype)
