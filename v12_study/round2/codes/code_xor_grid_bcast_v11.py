def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j. Bit decomposition, reduced ops per bit.
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)
    jj = idx.view(1, N)
    nbits = 1
    while (1 << nbits) < N:
        nbits += 1
    res = ((ii % 2) + (jj % 2)) % 2
    for b in range(1, nbits):
        p = 1 << b
        res = res + (((ii // p) + (jj // p)) % 2) * p
    return res.to(x.dtype)

