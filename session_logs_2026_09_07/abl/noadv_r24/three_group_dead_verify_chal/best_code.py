
def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute: max_r(x) + sum_r(y) + min_r(z)
    # max_r(x): global max of x across all ranks
    # sum_r(y): global sum of y across all ranks
    # min_r(z): global min of z across all ranks
    
    # Single all-reduce per operation (no redundant verification)
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    mz = xm.all_reduce(xm.REDUCE_MIN, z)
    
    return mx + sy + mz
