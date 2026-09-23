def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized all-reduce with reduced collective dispatch count.
    Reduces the number of iterations to minimize collective operations.
    """
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    
    # Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute global normalization factor and normalize
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iterative normalization: reduced to 3 rounds
    for i in range(3):
        # All-reduce the normalized buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        
        # For the last iteration, use acc directly as result
        if i == 2:
            s = acc
            break
        
        # Compute mean absolute value
        A = acc.abs().mean()
        
        # Compute multiplier to restore magnitude
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        
        # Scale up
        s = acc * gr
        
        # Normalize again
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    return s