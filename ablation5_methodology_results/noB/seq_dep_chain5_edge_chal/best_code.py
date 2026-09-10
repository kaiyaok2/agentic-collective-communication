def evolved_p3900(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    s = a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    s = s + 2 * a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    s = s + 3 * a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    s = s + 4 * a4
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5)
    s = s + 5 * a5
    return s
