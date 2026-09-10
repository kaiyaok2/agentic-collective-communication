def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    mz = xm.all_reduce(xm.REDUCE_MIN, z)
    return mx + sy + mz