
def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Single all-reduce after local sum
    s = x1 + x2 + x3 + x4 + x5
    result = xm.all_reduce(xm.REDUCE_SUM, s)
    return result
