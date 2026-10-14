def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    "per_mb_loop_AR: AR each layer's partial per-microbatch, mark_step between μbatches."
    for m in range(0, M, 2):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, partials[m][L]))
        outs.append(per_layer)
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, partials[m + 1][L]))
        outs.append(per_layer)
    return outs