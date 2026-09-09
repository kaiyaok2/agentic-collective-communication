
def evolved_p163(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = |i - j|
    # For fixed N=32, emit as constant tensor
    result = torch.tensor(
        [[abs(i - j) for j in range(N)] for i in range(N)],
        device=x.device,
        dtype=x.dtype
    )
    return result
