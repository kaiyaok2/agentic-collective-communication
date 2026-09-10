def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx_v = xm.all_reduce(xm.REDUCE_MAX, x)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    sy_v = xm.all_reduce(xm.REDUCE_SUM, sy)
    mz_v = xm.all_reduce(xm.REDUCE_MIN, z)
    return mx_v + sy_v / world_size + mz_v