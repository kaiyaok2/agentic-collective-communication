
def evolved_p3902(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = AR(x1) - AR(x2) + AR(x3) - AR(x4)
    # Mathematically equivalent: s = AR(x1 - x2 + x3 - x4)
    # This reduces 4 all-reduces to 1 all-reduce
    local_sum = x1 - x2 + x3 - x4
    s = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return s
