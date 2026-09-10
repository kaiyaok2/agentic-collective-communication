
def evolved_p4102(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first
    combined = 2.5 * x + 3.5 * y + 7.5 * z
    
    # Single all_reduce on the combined result
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    
    return result
