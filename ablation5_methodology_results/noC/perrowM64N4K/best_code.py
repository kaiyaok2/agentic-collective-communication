def perrowM64N4K_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce on entire tensor instead of per-row reduces
    # This is semantically equivalent but uses only 1 collective instead of M
    return xm.all_reduce(xm.REDUCE_SUM, x)