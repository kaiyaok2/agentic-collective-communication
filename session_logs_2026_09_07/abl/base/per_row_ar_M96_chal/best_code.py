
def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR full: all-reduce SUM on the entire tensor x (96, 512)
    # Single collective operation instead of 96 row-wise operations
    return xm.all_reduce(xm.REDUCE_SUM, x)
