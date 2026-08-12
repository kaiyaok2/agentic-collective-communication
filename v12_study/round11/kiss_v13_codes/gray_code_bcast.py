def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = i XOR (i >> 1) -- Gray code, position-based, shape (N,)
    vals = [i ^ (i >> 1) for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

