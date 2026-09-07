
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Reduce-scatter: sum x across all ranks, each rank gets its slice
    # Input: (world_size * N,) on each rank
    # Output: (N,) on each rank, containing rank's slice of the reduced sum
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size, groups=None)
