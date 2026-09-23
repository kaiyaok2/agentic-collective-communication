def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-reduce with scalar broadcasts strategy.
    Performs 8 all-reduce operations sequentially, where each iteration:
    1. Computes a global sum via all-reduce
    2. Derives a scalar normalization factor locally
    3. Applies the normalization
    4. Repeats for the next iteration
    
    This is the baseline approach with no fusion optimization.
    """
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # Iteration 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Iterations 2-7
    for _ in range(6):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
    
    # Final result
    s = acc
    return s