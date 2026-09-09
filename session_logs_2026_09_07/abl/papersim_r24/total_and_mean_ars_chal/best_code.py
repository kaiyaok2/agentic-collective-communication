
def evolved_p6701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute: y = AR(x) + AR(x)/W = AR(x) * (1 + 1/W)
    # Single all-reduce instead of two
    total = xm.all_reduce(xm.REDUCE_SUM, x)
    return total * (1.0 + 1.0 / world_size)
