def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = [[None] * N_LAYERS for _ in range(M)]
    for L in range(N_LAYERS):
        st = torch.stack([partials[m][L] for m in range(M)], dim=0)
        r = xm.all_reduce(xm.REDUCE_SUM, st)
        for m in range(M):
            outs[m][L] = r[m]
    return outs
