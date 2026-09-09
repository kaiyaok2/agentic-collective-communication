def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i XOR (i >> 1) - Gray code
    # Precompute Gray codes as constants (position-based, N=128 fixed)
    gray_codes = [i ^ (i >> 1) for i in range(N)]
    result = torch.tensor(gray_codes, device=x.device, dtype=x.dtype)
    return result