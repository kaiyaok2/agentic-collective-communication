def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with scalar normalization.
    Performs 8 dependent all-reduce operations sequentially, each followed by 
    local computation of mean absolute value and scalar division.
    """
    BETA = 0.7
    W = world_size
    dtype = x.dtype
    
    # Iteration 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 3
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 4
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 5
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 7
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iteration 8 (final)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s