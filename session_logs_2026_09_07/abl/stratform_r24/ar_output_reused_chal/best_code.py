
def evolved_p4800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single all_reduce with algebraic factorization
    ar_x = xm.all_reduce(xm.REDUCE_SUM, x)
    return 5 * ar_x
