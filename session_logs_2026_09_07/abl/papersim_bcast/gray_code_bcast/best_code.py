
def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i XOR (i >> 1) - Gray code at position i
    # Position-based: compute locally on each rank, no collective needed
    # Use Python bitwise ops and constant fold
    values = [i ^ (i >> 1) for i in range(N)]
    return torch.tensor(values, device=x.device, dtype=x.dtype)
