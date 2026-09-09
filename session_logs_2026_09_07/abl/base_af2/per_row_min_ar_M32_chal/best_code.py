def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: AR(MIN) full. Per-row MIN.
    # Input: x shape (32, 2048) on each rank
    # Output: element-wise MIN across all ranks
    # Single all-reduce on entire tensor instead of 32 separate calls
    return xm.all_reduce(xm.REDUCE_MIN, x)