def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    local_sum = x1 + x2 + x3 + x4
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result