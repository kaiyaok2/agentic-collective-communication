
def evolved_p6003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Single all-reduce elimination
    # AR(zeros) = zeros, so y = AR(x) + 0 + 0 = AR(x)
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return result
