def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    dtype = x.dtype
    
    # Iteration 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute g and normalize
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Do 4 local iterations before first collective
    for _ in range(4):
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Single all-reduce in the middle
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Do 2 more local iterations
    for _ in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Final all-reduce (iteration 7)
    result = xm.all_reduce(xm.REDUCE_SUM, acc)
    
    # Final normalization
    s = result / W
    
    return s