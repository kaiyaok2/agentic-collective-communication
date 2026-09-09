def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Perform a single all-reduce on the entire tensor
    # instead of 64 separate column-wise all-reduces
    return xm.all_reduce(xm.REDUCE_SUM, x)