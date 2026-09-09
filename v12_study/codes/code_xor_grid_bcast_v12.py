def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j. Vectorized bit decomposition.
    idx = torch.arange(N, device=x.device)
    nbits = max(1, (N - 1).bit_length())
    powers = torch.tensor([1 << b for b in range(nbits)], device=x.device)
    ai = (idx.view(N, 1, 1) // powers) % 2
    bj = (idx.view(1, N, 1) // powers) % 2
    out = (((ai + bj) % 2) * powers).sum(-1)
    return out.to(x.dtype)

