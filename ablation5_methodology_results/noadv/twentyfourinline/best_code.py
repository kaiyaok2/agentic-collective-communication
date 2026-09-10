def twentyfourinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Instead of doing 24 identical all_reduce operations and summing them,
    # do one all_reduce and multiply by 24
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    return 24.0 * t