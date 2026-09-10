def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    return a1 + a2 + a3 + a4
