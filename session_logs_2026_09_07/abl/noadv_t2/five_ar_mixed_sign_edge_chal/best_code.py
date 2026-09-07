
def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = AR(x1) - 0.5*AR(x2) + 2.5*AR(x3) - 1.5*AR(x4) + 3.0*AR(x5)
    # Since AR is linear (SUM), we can pre-combine locally:
    # s = AR(x1 - 0.5*x2 + 2.5*x3 - 1.5*x4 + 3.0*x5)
    local_sum = x1 - 0.5 * x2 + 2.5 * x3 - 1.5 * x4 + 3.0 * x5
    s = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return s
