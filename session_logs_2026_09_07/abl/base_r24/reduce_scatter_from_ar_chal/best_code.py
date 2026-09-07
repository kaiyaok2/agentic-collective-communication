def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: reduce SUM across all ranks, then each rank keeps its own slice
    # Input: x shape (world_size * N,) on each rank
    # Output: x[rank*N:(rank+1)*N] of the element-wise sum across all ranks
    # This is the definition of reduce_scatter!
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, 
                            shard_count=world_size, groups=None)