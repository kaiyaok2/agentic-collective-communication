def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum locally first
    local_sum = x1 + x2 + x3 + x4 + x5
    # Single all-reduce on the sum
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)