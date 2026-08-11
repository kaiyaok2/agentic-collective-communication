def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j.  XOR bit b = (i_b + j_b) mod 2.
    idx = torch.arange(N, device=x.device)
    nbits = 1
    while (1 << nbits) < N:
        nbits += 1
    powers = torch.tensor([1 << b for b in range(nbits)], device=x.device)
    ib = (idx.view(N, 1) // powers.view(1, nbits)) % 2
    xor = (ib.view(N, 1, nbits) + ib.view(1, N, nbits)) % 2
    out = (xor * powers.view(1, 1, nbits)).sum(-1)
    return out.to(x.dtype)

