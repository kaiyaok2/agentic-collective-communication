
def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first, then do single all_reduce
    # This is mathematically equivalent due to linearity of all_reduce
    s = x1 + 2 * x2 + 4 * x3 + 8 * x4
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
