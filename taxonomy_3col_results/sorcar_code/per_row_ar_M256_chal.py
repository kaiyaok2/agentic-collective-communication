def evolved_p6800(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on the entire tensor instead of per-row
    return xm.all_reduce(xm.REDUCE_SUM, x)