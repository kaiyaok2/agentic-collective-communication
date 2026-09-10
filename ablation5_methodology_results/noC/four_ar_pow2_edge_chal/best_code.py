
def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine tensors locally with their coefficients
    combined = x1 + 2 * x2 + 4 * x3 + 8 * x4
    # Single all_reduce on the combined result
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result
