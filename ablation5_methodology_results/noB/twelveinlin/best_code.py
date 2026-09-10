def twelveinlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Instead of 12 identical all_reduce operations,
    # do one all_reduce and multiply by 12
    return 12.0 * xm.all_reduce(xm.REDUCE_SUM, x)