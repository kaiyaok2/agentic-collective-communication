def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # reduce_scatter combines all_reduce + scatter in one efficient operation
    # Each rank gets its corresponding slice after the reduction
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)