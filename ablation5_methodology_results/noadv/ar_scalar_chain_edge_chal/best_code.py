
def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute linear combination locally, then all-reduce once
    # This is mathematically equivalent to:
    # 5 * all_reduce(z) + 3 * all_reduce(y) + 2 * all_reduce(x)
    # But reduces 3 collectives to 1
    combined = 2 * x + 3 * y + 5 * z
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result
