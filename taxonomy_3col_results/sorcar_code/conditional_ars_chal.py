def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum locally first
    local_sum = x + y + z
    # Single all_reduce instead of three
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result