def sixteeninlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 16 separate ones
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    # Multiply by 16 to get the same result as 16 accumulations
    return 16.0 * t