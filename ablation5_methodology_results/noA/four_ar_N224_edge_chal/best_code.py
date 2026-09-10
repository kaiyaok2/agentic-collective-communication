
def evolved_p4402(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine locally before all_reduce using linearity property
    # all_reduce(2*x1) + all_reduce(3*x2) + ... = all_reduce(2*x1 + 3*x2 + ...)
    local_sum = 2 * x1 + 3 * x2 + 5 * x3 + 7 * x4
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
