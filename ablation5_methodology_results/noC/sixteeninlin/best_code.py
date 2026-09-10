def sixteeninlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce multiplied by 16
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    return t * 16.0