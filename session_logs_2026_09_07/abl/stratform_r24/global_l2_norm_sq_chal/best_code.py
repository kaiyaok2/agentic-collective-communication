
def evolved_p6100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Local sum first, then all-reduce scalar
    sq = x * x
    local_sum = sq.sum()
    global_sum = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return global_sum
