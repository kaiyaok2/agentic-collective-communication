def evolved_p6203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return 5 * xm.all_reduce(xm.REDUCE_SUM, x)