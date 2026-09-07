
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Reduce SUM across all ranks, each rank gets only its own slice
    # reduce_scatter: input (world_size*N,), output (N,) per rank
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)
