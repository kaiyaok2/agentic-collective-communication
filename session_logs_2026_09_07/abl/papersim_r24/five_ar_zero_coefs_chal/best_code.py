
def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 3*AR(x1) + 5*AR(x3) + 2*AR(x5)
    # Optimization: AR(3*x1 + 5*x3 + 2*x5) using distributivity
    # Single all_reduce instead of 3-5 separate ones
    local_sum = 3 * x1 + 5 * x3 + 2 * x5
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
