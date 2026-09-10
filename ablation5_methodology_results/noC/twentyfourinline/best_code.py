def twentyfourinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce followed by scalar multiplication
    # This is equivalent to summing 24 identical all_reduce results
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    return 24.0 * t