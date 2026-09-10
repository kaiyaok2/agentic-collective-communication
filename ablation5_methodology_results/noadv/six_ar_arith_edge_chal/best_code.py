
def evolved_p4702(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Weight locally first, then single all_reduce
    weighted = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 6*x6
    s = xm.all_reduce(xm.REDUCE_SUM, weighted)
    return s
