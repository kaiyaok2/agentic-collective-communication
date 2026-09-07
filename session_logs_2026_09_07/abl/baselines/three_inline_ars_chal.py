def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    b = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    c = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a + b + c
