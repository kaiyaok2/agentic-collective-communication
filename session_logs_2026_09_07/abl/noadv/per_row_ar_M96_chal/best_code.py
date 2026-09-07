
def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: AR full = all-reduce SUM on entire tensor x (96, 512)
    # Output: same shape, each element summed across all ranks
    # Optimization: single all-reduce instead of 96 row-by-row calls
    return xm.all_reduce(xm.REDUCE_SUM, x)
