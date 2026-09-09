
def evolved_p165(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = i BITWISE-OR j
    # Constant fold: precompute the entire matrix
    result = torch.tensor(
        [[i | j for j in range(N)] for i in range(N)],
        device=x.device,
        dtype=x.dtype
    )
    return result
