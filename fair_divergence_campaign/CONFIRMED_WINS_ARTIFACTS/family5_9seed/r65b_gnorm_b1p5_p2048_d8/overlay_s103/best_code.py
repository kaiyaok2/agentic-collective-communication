def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential All-Reduce Chain (Baseline)
    
    Direct translation of the reference implementation:
    - 8 separate all-reduce dispatches
    - Interleaved local computation (abs, mean, division)
    - Total: 8 collective dispatches
    - Naive approach with maximum dispatch overhead but simplest correctness
    """
    W = world_size
    BETA = 1.5
    dtype = x.dtype
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Iterations 2-7 (6 more all-reduces)
    for _ in range(6):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
    
    s = acc
    return s