
def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR full: all-reduce sum across all ranks for the full tensor x
    # Single all-reduce instead of 1024 row-by-row operations
    return xm.all_reduce(xm.REDUCE_SUM, x)
