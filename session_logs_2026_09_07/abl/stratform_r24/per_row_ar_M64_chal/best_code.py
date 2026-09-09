
def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 4: Reduce-scatter + all-gather pattern
    # Reduce-scatter: each rank gets a portion of reduced rows
    scattered = xm.reduce_scatter(xm.REDUCE_SUM, x.flatten(), scale=1.0, scatter_dim=0, shard_count=world_size)
    # All-gather: collect all portions
    gathered = xm.all_gather(scattered, dim=0)
    return gathered.reshape(x.shape)
