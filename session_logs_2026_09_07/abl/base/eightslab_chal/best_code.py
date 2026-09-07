
def evolved_p9002(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: SUM_reduce_over_ranks(x)
    # Compute element-wise sum of x across all ranks
    # Return (8192,) identical on every rank
    
    # Single all-reduce - should minimize dispatch overhead
    return xm.all_reduce(xm.REDUCE_SUM, x)
