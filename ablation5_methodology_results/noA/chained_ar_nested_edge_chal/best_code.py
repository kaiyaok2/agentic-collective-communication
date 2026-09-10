
def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try using reduce_scatter for the first reduction
    shard = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)
    # Gather it back
    ax1 = xm.all_gather(shard.unsqueeze(0), dim=0).view(-1)
    # Continue with the rest
    ax2 = xm.all_reduce(xm.REDUCE_SUM, x + ax1 * 2)
    return xm.all_reduce(xm.REDUCE_SUM, x + ax2 + 3)
