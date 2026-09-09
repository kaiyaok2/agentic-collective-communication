def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # AR full: all-reduce the entire tensor in one collective call
    # Input: x shape (1024, 64) local to each rank
    # Output: sum of x across all ranks, shape (1024, 64)
    return xm.all_reduce(xm.REDUCE_SUM, x)