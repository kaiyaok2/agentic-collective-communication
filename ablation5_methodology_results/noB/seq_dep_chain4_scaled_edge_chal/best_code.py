def evolved_p3901(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = 1.5 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 2.5 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 3.5 * az
    aw = xm.all_reduce(xm.REDUCE_SUM, w)
    s = s + 4.5 * aw
    return s
