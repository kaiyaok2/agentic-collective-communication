def evolved_p6202(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute local combination first (element-wise operations)
    local_result = x1 - x2 + x3 - x4 + x5
    # Single all_reduce on the combined result
    return xm.all_reduce(xm.REDUCE_SUM, local_result)