def mixmaxmin_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    max_val = xm.all_reduce(xm.REDUCE_MAX, x)
    min_val = xm.all_reduce(xm.REDUCE_MIN, x)
    return 3.6 * max_val + 1.8 * min_val