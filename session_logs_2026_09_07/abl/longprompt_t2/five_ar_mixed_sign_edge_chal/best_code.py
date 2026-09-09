
def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute linear combination locally first
    # s = AR(x1) - 0.5*AR(x2) + 2.5*AR(x3) - 1.5*AR(x4) + 3*AR(x5)
    # = AR(x1 - 0.5*x2 + 2.5*x3 - 1.5*x4 + 3*x5)
    temp = x1 - 0.5 * x2 + 2.5 * x3 - 1.5 * x4 + 3.0 * x5
    s = xm.all_reduce(xm.REDUCE_SUM, temp)
    return s
