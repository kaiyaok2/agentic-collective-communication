def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        g = xm.all_gather(st, dim=-1)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
