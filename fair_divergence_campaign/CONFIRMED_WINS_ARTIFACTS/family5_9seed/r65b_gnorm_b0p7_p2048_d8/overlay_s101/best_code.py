def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    dtype = x.dtype
    
    # Iteration 0: Initial all-reduce of x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-2: Further reduced loop count (only 2 iterations)
    for iteration in range(2):
        # Compute normalization for current s
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Async dispatch all-reduce (queued immediately)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Continue with local computation while communication may overlap
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Final iteration (3rd all-reduce): Just normalize and reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc