def perrowM2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Reduce entire tensor at once instead of row-by-row
    return xm.all_reduce(xm.REDUCE_SUM, x)