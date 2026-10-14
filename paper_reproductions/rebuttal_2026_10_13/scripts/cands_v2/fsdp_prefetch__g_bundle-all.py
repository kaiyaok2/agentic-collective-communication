def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([shards[m][L] for m in range(M) for L in range(N_LAYERS)], dim=0)
    g = xm.all_gather(st, dim=-1)
    return [[g[m * N_LAYERS + L] for L in range(N_LAYERS)] for m in range(M)]
