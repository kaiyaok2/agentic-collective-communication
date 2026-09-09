
def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Local weighted sum then single all-reduce
    local_sum = 3 * x1 + 5 * x3 + 2 * x5
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
