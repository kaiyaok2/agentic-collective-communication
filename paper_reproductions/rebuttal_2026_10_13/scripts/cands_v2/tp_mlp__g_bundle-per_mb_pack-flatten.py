def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(partials[m], dim=0)
        r = xm.all_reduce(xm.REDUCE_SUM, st)
        outs.append([r[L] for L in range(N_LAYERS)])
    return outs
