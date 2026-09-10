def evolved_p4402(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 7 * a
    return s
