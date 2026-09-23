def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    
    # Initial reduction - only all_reduce needed
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # All 6 iterations without all_reduce (s is synchronized after initial all_reduce)
    for _ in range(6):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = buf  # No all_reduce needed - s is same on all ranks
        A = acc.abs().mean()
        gr = 1.0 / (1.0 - BETA * A)
        s = acc * gr
    
    # Final normalization
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    
    return s