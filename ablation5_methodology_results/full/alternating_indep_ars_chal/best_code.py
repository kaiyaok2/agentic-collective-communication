
def evolved_p6202(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    combined = x1 - x2 + x3 - x4 + x5
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result
