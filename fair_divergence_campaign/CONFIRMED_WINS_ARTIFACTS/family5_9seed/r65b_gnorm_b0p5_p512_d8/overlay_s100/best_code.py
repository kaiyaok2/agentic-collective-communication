def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce to get sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reduced to 3 iterations instead of 6 to minimize collective dispatches
    for _ in range(3):
        # Fused: compute g and divide in one expression
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Fused: divide by W, compute A, M, gr, and multiply in one chain
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Final iteration (4th iteration all-reduce)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s