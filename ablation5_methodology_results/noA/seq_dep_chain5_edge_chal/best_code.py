
def evolved_p3900(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    s = x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5
    # Single all_reduce on the weighted sum
    result = xm.all_reduce(xm.REDUCE_SUM, s)
    return result
