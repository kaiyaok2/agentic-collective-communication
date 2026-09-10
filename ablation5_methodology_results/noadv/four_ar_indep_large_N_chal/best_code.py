
def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Sum locally first, then all_reduce once
    # This exploits: all_reduce(x1) + all_reduce(x2) + ... = all_reduce(x1 + x2 + ...)
    local_sum = x1 + x2 + x3 + x4
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
