
def evolved_p6003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Eliminate dead code - just AR(x)
    y = xm.all_reduce(xm.REDUCE_SUM, x)
    return y
