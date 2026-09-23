def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with iterative refinement.
    Performs 8 dependent all-reduce operations with local mean computations.
    """
    W = world_size
    BETA = 1.0
    
    # Iteration 1: Initial all-reduce of input
    s = xm.all_reduce(xm.REDUCE_SUM, x)
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