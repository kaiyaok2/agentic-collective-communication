def twentyinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 20
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    # Multiply by 20 to get the same accumulated result
    result = 20.0 * reduced
    return result