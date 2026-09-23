def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce: sum of x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Unroll the loop to minimize dispatch overhead
    # Do 5 full iterations, then 1 final simplified iteration
    
    for iteration in range(5):
        # Compute g = 1.0 + BETA * s.abs().mean()
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # All-reduce buf
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        
        # Compute next s
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Final iteration (6th): no need to compute A, M, gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return acc