
def evolved_p4402(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    s = 2 * x1 + 3 * x2 + 5 * x3 + 7 * x4
    # Then perform single all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
