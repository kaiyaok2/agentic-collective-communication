
def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    s = 3 * x1 + 0.5 * x2 + 7 * x3 + 1.5 * x4
    # Then all_reduce once
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
