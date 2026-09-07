
def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all-reduce on the entire tensor instead of 1024 separate operations
    return xm.all_reduce(xm.REDUCE_SUM, x)
