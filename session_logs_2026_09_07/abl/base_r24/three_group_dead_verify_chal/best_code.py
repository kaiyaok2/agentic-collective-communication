
def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute max_r(x) + sum_r(y) + min_r(z)
    # Using only 3 all-reduce operations (no verification)
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    mz = xm.all_reduce(xm.REDUCE_MIN, z)
    return mx + sy + mz
