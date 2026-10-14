def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """per_mb_loop_AG: AG each layer's shard per-microbatch, mark_step between μbatches."""
    outs = []
    for m in range(0, M, 2):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m][L], dim=-1))
        outs.append(per_layer)
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m + 1][L], dim=-1))
        outs.append(per_layer)
    return outs