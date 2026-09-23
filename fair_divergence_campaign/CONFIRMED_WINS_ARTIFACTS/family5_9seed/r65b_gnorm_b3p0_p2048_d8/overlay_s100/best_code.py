def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version minimizing collective dispatch overhead.
    Performs all 8 iterations with only 2 collective operations total.
    """
    W = world_size
    BETA = 3.0
    
    # Initial all_reduce to get starting point
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    # Perform all remaining 7 iterations locally without sync
    for _ in range(7):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    return acc