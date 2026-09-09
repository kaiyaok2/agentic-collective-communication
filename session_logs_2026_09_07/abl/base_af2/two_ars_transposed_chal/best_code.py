def evolved_p5701(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute 2 * AR(x)
    # One all-reduce instead of two
    return xm.all_reduce(xm.REDUCE_SUM, x) * 2