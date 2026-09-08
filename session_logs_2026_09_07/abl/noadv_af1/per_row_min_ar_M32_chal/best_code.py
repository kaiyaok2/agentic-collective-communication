
def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: Local x (32, 2048). AR(MIN) full. Per-row MIN.
    # Compute element-wise minimum of x across all ranks for each position.
    # Output shape: (32, 2048)
    
    # Single all-reduce MIN on the entire tensor instead of 32 separate calls
    result = xm.all_reduce(xm.REDUCE_MIN, x)
    return result
