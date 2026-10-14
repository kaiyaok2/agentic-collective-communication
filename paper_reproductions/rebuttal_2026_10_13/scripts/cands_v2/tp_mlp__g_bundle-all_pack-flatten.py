def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    shapes = [partials[m][L].shape for m in range(M) for L in range(N_LAYERS)]
    flat = torch.cat([partials[m][L].flatten() for m in range(M) for L in range(N_LAYERS)], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for m in range(M):
        row = []
        for L in range(N_LAYERS):
            n = 1
            for d in shapes[m * N_LAYERS + L]: n *= d
            row.append(r[off:off + n].reshape(shapes[m * N_LAYERS + L])); off += n
        outs.append(row)
    return outs
