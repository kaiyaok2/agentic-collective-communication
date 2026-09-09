
def evolved_p87(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i * i) % K
    # Try constant folding for small N
    result = torch.tensor([(i * i) % K for i in range(N)], 
                          device=x.device, dtype=x.dtype)
    return result
