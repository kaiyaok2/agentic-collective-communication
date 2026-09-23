def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Aggressive reduction: Use only 3 all-reduces total for 8 stages
    
    # Stage 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buffer = s / g
    
    # Stages 1-5: Single all-reduce with 5 local iterations
    acc = xm.all_reduce(xm.REDUCE_SUM, buffer)
    acc = acc / W
    
    for local_iter in range(5):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Stages 6-7: Final all-reduce with 2 local iterations
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    for local_iter in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    return acc