
def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Single all-reduce
    result = xm.all_reduce(xm.REDUCE_SUM, x) * 15
    return result
