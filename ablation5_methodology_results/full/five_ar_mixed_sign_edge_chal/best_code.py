
def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Apply coefficients locally and sum
    local_sum = x1 - 0.5 * x2 + 2.5 * x3 - 1.5 * x4 + 3.0 * x5
    # Single all_reduce instead of 5 separate ones
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
