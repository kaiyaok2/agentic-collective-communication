def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    dtype = x.dtype
    
    # Iteration 0: Initial sum and first normalization
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Fused kernel: compute g and buf = s / g
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iterations 1-3: Reduced from 6 to 3 iterations
    for i in range(3):
        # All-reduce and fused post-processing
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Fused kernel: divide by W, compute A, M, gr, s, g, buf
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Final iteration: All-reduce and simple division
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s