
def evolved_p4702(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first (essentially free operations)
    local_sum = x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6
    # Single all_reduce instead of 6 separate ones
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
