def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce: sum of x across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused kernel: compute g, divide s by g
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Loop for iterations 2-7 (6 iterations)
    for i in range(6):
        # All-reduce: sum of buf
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Fused kernel: divide by W, compute A, M, gr, s, g, buf
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Final all-reduce (iteration 8)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Fused kernel: final division by W
    s = acc / W
    
    return s