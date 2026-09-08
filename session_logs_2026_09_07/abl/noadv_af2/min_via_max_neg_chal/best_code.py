def evolved_p5603(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute element-wise MIN across all ranks
    # Direct MIN reduction instead of -MAX(-x) trick
    return xm.all_reduce(xm.REDUCE_MIN, x)