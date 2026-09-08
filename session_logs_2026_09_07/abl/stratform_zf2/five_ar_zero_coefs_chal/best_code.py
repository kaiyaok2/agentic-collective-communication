
def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single all-reduce of pre-computed weighted sum
    local_sum = 3 * x1 + 5 * x3 + 2 * x5
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
