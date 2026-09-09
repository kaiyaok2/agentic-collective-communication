def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = AR(x) + 2*AR(x) + 3*AR(x) = 6*AR(x)
    # Optimize: call all_reduce once instead of three times
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return 6 * ar_x