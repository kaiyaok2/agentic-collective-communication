def evolved_p5000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce to compute global maximum
    m = xm.all_reduce(xm.REDUCE_MAX, x)
    return m