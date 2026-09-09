def evolved_p9002(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: SUMreduce_over_ranks(x) - element-wise sum of x across all ranks
    # Single all_reduce is more efficient than splitting into 8 parts
    return xm.all_reduce(xm.REDUCE_SUM, x)