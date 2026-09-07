
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Direct reduce-scatter
    result = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1, scatter_dim=0, shard_count=world_size)
    return result
