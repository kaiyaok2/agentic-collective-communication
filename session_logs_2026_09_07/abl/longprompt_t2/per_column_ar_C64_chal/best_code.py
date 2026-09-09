
def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All-reduce the entire tensor at once instead of column-by-column
    # This reduces 64 all_reduce calls to just 1
    return xm.all_reduce(xm.REDUCE_SUM, x)
