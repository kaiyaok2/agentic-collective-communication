def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce the entire tensor at once instead of column-by-column
    # Each element will be summed across all ranks, which is exactly
    # what we want for "AR full"
    return xm.all_reduce(xm.REDUCE_SUM, x)