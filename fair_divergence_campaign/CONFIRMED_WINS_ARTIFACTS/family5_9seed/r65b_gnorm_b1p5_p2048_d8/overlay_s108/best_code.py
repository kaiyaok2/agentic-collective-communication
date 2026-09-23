def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version with minimal collective operations.
    
    Strategy: Maximize local computation by doing many refinement iterations
    locally before synchronizing. Use only 3 total all-reduces to minimize
    collective dispatch overhead.
    """
    W = world_size
    BETA = 1.5
    dtype = x.dtype
    
    # Stage 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First batch: 5 local iterations to amortize collective cost
    for _ in range(5):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
    
    # Synchronize after first batch
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    s = acc / W
    
    # Second batch: 5 local iterations
    for _ in range(5):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
    
    # Final normalization and all-reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Final all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s