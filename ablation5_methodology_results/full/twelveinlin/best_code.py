def twelveinlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 12, then multiply by 12
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    result = 12.0 * result
    return result