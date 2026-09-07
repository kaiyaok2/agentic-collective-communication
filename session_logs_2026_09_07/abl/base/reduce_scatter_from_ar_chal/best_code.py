
def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: Reduce SUM across all ranks, each rank keeps slice [rank*N:(rank+1)*N]
    # Use reduce_scatter: splits input into world_size chunks, reduces each, 
    # and rank i receives the i-th reduced chunk
    result = xm.reduce_scatter(
        xm.REDUCE_SUM, 
        x, 
        scale=1.0, 
        scatter_dim=0, 
        shard_count=world_size,
        groups=None
    )
    return result
