
def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine local computations first: 2*x + 3*y + 5*z
    # Then do a single all_reduce instead of 3 separate ones
    # This is mathematically equivalent due to linearity of sum reduction
    combined = 2 * x + 3 * y + 5 * z
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result
