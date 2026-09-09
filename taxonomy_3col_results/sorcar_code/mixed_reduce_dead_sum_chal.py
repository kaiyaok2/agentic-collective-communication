def evolved_p5203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    m = xm.all_reduce(xm.REDUCE_MAX, x)
    return s + m