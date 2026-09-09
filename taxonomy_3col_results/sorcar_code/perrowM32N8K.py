def perrowM32N8K_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-row
    # all_reduce is element-wise, so reducing (M,K) gives same result 
    # as reducing each row and stacking
    return xm.all_reduce(xm.REDUCE_SUM, x)