def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [[xm.all_reduce(xm.REDUCE_SUM, partials[m][L]) for L in range(N_LAYERS)] for m in range(M)]
