def tenariindep_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 10
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    return 10.0 * t