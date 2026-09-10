
def evolved_p4300(x1, x2, x3, x4, x5, x6, x7, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    s = x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7
    # Then single all_reduce to get global sum
    result = xm.all_reduce(xm.REDUCE_SUM, s)
    return result
