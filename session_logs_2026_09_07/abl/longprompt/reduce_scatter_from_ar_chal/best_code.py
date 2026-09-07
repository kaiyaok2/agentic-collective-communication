
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Use reduce_scatter instead of all_reduce + narrow
    # This reduces the tensor and each rank gets its slice directly
    # Input: (world_size * N,), Output: (N,)
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)
