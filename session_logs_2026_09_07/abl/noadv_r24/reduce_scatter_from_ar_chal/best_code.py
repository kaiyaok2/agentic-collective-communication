
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: reduce SUM across all ranks, then each rank keeps only its own slice
    # Input: x has shape (world_size * N,)
    # Output: reduced[rank*N:(rank+1)*N] of shape (N,)
    
    # reduce_scatter: reduces and scatters in one operation
    # Each rank gets its portion of the reduced result
    # More efficient than all_reduce + narrow (transfers N instead of world_size*N per rank)
    result = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size, groups=None)
    return result
