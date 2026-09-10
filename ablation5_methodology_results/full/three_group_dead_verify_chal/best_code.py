
def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce for MAX
    mx_v = xm.all_reduce(xm.REDUCE_MAX, x)
    
    # The double reduction pattern: sum, then sum again / world_size
    # Equivalent to: world_size * sum / world_size = sum
    # So we just need single sum
    sy_v = xm.all_reduce(xm.REDUCE_SUM, y)
    
    # Single all_reduce for MIN  
    mz_v = xm.all_reduce(xm.REDUCE_MIN, z)
    
    return mx_v + sy_v + mz_v
