def mixmaxmin_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    max_x = xm.all_reduce(xm.REDUCE_MAX, x)
    min_x = xm.all_reduce(xm.REDUCE_MIN, x)
    result = 3.6 * max_x + 1.8 * min_x
    return result