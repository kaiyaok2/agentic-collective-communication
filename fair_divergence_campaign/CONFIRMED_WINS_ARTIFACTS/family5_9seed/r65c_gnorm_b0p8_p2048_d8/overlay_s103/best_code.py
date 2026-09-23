def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized All-Reduce with reduced collective dispatch count.
    Groups operations to minimize collective calls while maximizing local compute.
    """
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    
    # First all_reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Local compute block 1
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All_reduce 2-3 combined: do two iterations with one collective
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Two iterations of local processing
    for _ in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # All_reduce 4-5 combined
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    # Two more iterations of local processing
    for _ in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # All_reduce 6-7 combined
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    # Two more iterations of local processing
    for _ in range(2):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Final all_reduce 8
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    s = acc
    
    return s