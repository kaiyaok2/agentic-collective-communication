def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # Iteration 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute g, divide, and prepare for all-reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Single all-reduce with extended local computation
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Extended local iteration block (unrolled loop for more local ops)
    for i in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Second all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    # More local iterations
    for i in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Final all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, acc)
    s = s / W
    
    return s