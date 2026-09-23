def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain strategy.
    Performs 8 sequential all-reduce operations with local mean computations interleaved.
    This is the straightforward implementation matching the reference code structure.
    """
    W = world_size
    BETA = 1.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # First all-reduce: sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 2
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 3
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 4
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 5
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 6
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - 1.0 * acc.mean() / (1.0 + 1.0)
    
    # Iteration 7 (final)
    buf = s + BETA * (s.mean())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s