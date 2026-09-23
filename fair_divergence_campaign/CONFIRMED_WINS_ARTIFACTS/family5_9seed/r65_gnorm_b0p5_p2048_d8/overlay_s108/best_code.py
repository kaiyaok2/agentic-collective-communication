def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with local normalization.
    Performs 8 sequential all-reduce operations with local normalization steps between them.
    """
    W = world_size
    BETA = 0.5
    
    # Preserve input dtype
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 3
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 4
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 5
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 7
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 8 (final)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s