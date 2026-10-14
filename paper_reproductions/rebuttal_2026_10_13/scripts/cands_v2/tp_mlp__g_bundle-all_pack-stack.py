def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([partials[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, st)
    return [[r[m * N_LAYERS + L] for L in range(N_LAYERS)] for m in range(M)]
