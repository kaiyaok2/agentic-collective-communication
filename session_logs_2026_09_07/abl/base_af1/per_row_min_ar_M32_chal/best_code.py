
def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: Local x (32, 2048). AR(MIN) full. Per-row MIN.
    # Paraphrase: For each position (i, j), compute the minimum of x[i, j]
    # across all ranks using all-reduce with MIN operation.
    # Output shape: (32, 2048)
    
    # Single all-reduce on the entire tensor instead of 32 separate ones
    return xm.all_reduce(xm.REDUCE_MIN, x)
