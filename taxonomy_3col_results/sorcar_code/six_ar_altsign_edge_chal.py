
def evolved_p4502(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute linear combination locally first
    s = 3 * x1 - 2 * x2 + 5 * x3 - 4 * x4 + 7 * x5 - 6 * x6
    # Then do single all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, s)
    return result
