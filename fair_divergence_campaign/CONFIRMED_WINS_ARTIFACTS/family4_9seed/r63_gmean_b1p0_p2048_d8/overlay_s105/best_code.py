def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Reduced collective dispatch strategy.
    
    Minimizes the number of all-reduce operations by doing more local
    computation between collectives. Groups iterations to reduce dispatch overhead.
    """
    W = world_size
    BETA = 1.0
    dtype = x.dtype
    
    # First all-reduce: sum input x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Do 3 iterations locally, then sync
    for _ in range(3):
        buf = s + BETA * s.mean()
        s = buf - buf.mean() / 2.0
    
    # All-reduce after local iterations
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    s = acc / W
    
    # Do 3 more iterations locally, then sync
    for _ in range(3):
        buf = s + BETA * s.mean()
        s = buf - buf.mean() / 2.0
    
    # All-reduce after local iterations
    acc = xm.all_reduce(xm.REDUCE_SUM, s)
    s = acc / W
    
    # Final iteration
    buf = s + BETA * s.mean()
    
    # Final all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s