def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with scalar mean broadcast.
    Performs 8 sequential all-reduce operations where each iteration:
    1. Computes local mean of current tensor
    2. Adds BETA * mean to the tensor
    3. All-reduces the result
    4. Divides by world_size
    5. Subtracts correction term (except last iteration)
    
    This follows the reference implementation exactly.
    """
    W = world_size
    BETA = 1.0
    
    # First all-reduce: sum x across all ranks
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