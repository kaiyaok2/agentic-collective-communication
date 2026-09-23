def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Unroll and combine iterations to reduce collectives
    # Do 3 double-iterations instead of 6 single iterations
    for _ in range(3):
        # First sub-iteration (local)
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc_local = s / g
        
        # Second sub-iteration (local)
        A = acc_local.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc_local * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Single all-reduce for both iterations
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
    
    s = acc
    return s