def evolved_p6302(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute 4 * all_reduce(x) with a single all_reduce
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    return 4 * a