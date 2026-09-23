def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential All-Reduce Chain (Baseline)
    
    Performs 8 sequential all-reduce operations with interleaved local mean 
    computations and scalar broadcasts. Each all-reduce is dispatched separately 
    (8 total dispatches). This is the naive approach with maximum dispatch 
    overhead but simplest correctness verification.
    """
    W = world_size
    BETA = 1.0
    
    # Preserve input dtype
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
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