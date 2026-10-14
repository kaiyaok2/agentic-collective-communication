def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [[xm.all_gather(shards[m][L], dim=-1) for L in range(N_LAYERS)] for m in range(M)]
