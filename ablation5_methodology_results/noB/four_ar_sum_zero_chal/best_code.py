def evolved_p6402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    b = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    c = xm.all_reduce(xm.REDUCE_SUM, x)
    d = 4 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a + b - c - d
