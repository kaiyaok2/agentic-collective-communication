
def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    s_local = x1 + 2 * x2 + 4 * x3 + 8 * x4
    # Then do a single all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s_local)
    return s
