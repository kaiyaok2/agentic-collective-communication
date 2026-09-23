def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    dtype = x.dtype
    
    # First all-reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local computation: iterations 1-4 without intermediate all-reduces
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Local iterations 2-4
    for i in range(3):
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Second all-reduce after local iterations
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # More local iterations 5-7
    for i in range(3):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Final all-reduce (iteration 8)
    result = xm.all_reduce(xm.REDUCE_SUM, acc)
    s = result / W
    
    return s