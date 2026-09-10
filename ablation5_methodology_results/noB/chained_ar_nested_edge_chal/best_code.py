
def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of three
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute the coefficient: 1 + world_size + 2 * world_size^2
    coefficient = 1 + world_size + 2 * world_size * world_size
    
    # Final result with local arithmetic only
    result = ax1 * coefficient + 3 * world_size
    
    return result
