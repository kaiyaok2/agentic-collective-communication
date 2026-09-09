
def evolved_p4400(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first (avoid 1.0 multiplication)
    weighted = 0.5 * x1 + x2 + 1.5 * x3 + 2.0 * x4 + 2.5 * x5
    # Single all_reduce on the weighted sum
    result = xm.all_reduce(xm.REDUCE_SUM, weighted)
    return result
