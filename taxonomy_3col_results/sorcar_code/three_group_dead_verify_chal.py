def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx_v = xm.all_reduce(xm.REDUCE_MAX, x)
    mz_v = xm.all_reduce(xm.REDUCE_MIN, z)
    
    # For sum: try reduce_scatter + all_gather pattern
    # This might be more efficient than double all_reduce
    shard = xm.reduce_scatter(xm.REDUCE_SUM, y, scale=1.0, scatter_dim=0, shard_count=world_size)
    sy_v = xm.all_gather(shard.unsqueeze(0), dim=0, groups=[list(range(world_size))]).flatten()
    
    return mx_v + sy_v + mz_v