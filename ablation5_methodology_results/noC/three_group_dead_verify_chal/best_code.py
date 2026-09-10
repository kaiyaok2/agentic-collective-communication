def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    mx_v = xm.all_reduce(xm.REDUCE_MAX, mx)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    sy_v = xm.all_reduce(xm.REDUCE_SUM, sy) / world_size
    mz = xm.all_reduce(xm.REDUCE_MIN, z)
    mz_v = xm.all_reduce(xm.REDUCE_MIN, mz)
    return mx_v + sy_v + mz_v
