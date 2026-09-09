def evolved_p5603(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise MIN across all ranks
    # Direct REDUCE_MIN should be faster than -MAX(-x) trick
    return xm.all_reduce(xm.REDUCE_MIN, x)