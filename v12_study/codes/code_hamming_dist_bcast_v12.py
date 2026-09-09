def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i,j] = popcount(i XOR j) = sum over bits of (ib XOR jb) = (ib+jb)%2
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)
    jj = idx.view(1, N)
    pc = (ii + jj) % 2
    for b in (2, 4, 8):
        pc = pc + (((ii // b) + (jj // b)) % 2)
    return pc.to(x.dtype)

