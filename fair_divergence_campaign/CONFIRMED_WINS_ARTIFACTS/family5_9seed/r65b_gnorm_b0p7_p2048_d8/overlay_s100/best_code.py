def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential All-Reduce with Scalar Broadcasts strategy.
    
    Performs 8 iterations of:
    1. All-reduce to compute sum across ranks
    2. Compute global normalization factor using mean absolute value
    3. Normalize by the factor
    
    For iterations 1-6: also compute and use adjustment factor M
    For iterations 7-8: just compute the final result
    """
    W = world_size
    BETA = 0.7
    dtype = x.dtype
    
    # Iteration 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    mean_abs = s.abs().mean()
    # Broadcast mean_abs via scalar all-reduce (already global after vector all-reduce)
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    mean_abs = s.abs().mean()
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 3
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    mean_abs = s.abs().mean()
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 4
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    mean_abs = s.abs().mean()
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 5
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    mean_abs = s.abs().mean()
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    mean_abs = s.abs().mean()
    g = 1.0 + BETA * mean_abs
    buf = s / g
    
    # Iteration 7
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s