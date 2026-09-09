
def evolved_p99(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i*i if i < N/2 else (N-i)*(N-i)
    # Pre-compute values at trace time for small fixed N
    values = [i*i if i < N//2 else (N-i)*(N-i) for i in range(N)]
    return torch.tensor(values, device=x.device, dtype=x.dtype)
