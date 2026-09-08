
def evolved_p6103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = 4 * AR(x)
    # where AR is all-reduce sum
    # Input: x is (N,) local on each rank
    # Output: (N,) identical on every rank
    
    # All-reduce sum across all ranks
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Multiply by 4
    y = 4 * ax
    
    return y
