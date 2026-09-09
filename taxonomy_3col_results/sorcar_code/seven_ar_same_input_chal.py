def evolved_p6300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine 7 all_reduce ops into 1
    # Original: (1 + 5 + 10 + 15 + 20 + 25 + 24) = 100
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 100 * reduced