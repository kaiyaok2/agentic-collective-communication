def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The alternating sum 1 - 2 + 3 - 4 + ... + 31 - 32 = -16
    return -16.0 * xm.all_reduce(xm.REDUCE_SUM, x)