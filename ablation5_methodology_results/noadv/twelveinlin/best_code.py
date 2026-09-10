def twelveinlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # All 12 all_reduce operations are on the same input x
    # So the sum of 12 all_reduce results is just 12 * all_reduce(x)
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 12.0 * result