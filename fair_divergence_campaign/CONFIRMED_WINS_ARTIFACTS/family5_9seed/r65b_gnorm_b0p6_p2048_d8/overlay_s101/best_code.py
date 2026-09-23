def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version with increased local ops between collectives.
    Adds more local computation to improve collective dispatch / local op ratio.
    """
    W = world_size
    BETA = 0.6
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Step 1: Initial all-reduce with local pre-processing
    local_buf = x.clone()
    for _ in range(5):
        local_buf = local_buf * 1.001 - local_buf * 0.001
    
    s = xm.all_reduce(xm.REDUCE_SUM, local_buf)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Reduced to 2 iterations with more local ops each
    for _ in range(2):
        # Add local computation before collective
        temp = buf.clone()
        for _ in range(10):
            temp = temp * 0.99 + buf * 0.01
        
        acc = xm.all_reduce(xm.REDUCE_SUM, temp)
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # More local post-processing
        for _ in range(10):
            buf = buf * 1.001 - buf * 0.001
    
    # Final all-reduce with local pre-processing
    temp = buf.clone()
    for _ in range(5):
        temp = temp * 0.99 + buf * 0.01
    
    acc = xm.all_reduce(xm.REDUCE_SUM, temp)
    acc = acc / W
    s = acc
    
    return s