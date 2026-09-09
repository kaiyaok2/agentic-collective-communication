
def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Use reduce_scatter followed by all_gather
    shard = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)
    sum_x = xm.all_gather(shard.unsqueeze(0), dim=0).flatten()
    contrib = y + sum_x * 2  
    return xm.all_reduce(xm.REDUCE_SUM, contrib)
