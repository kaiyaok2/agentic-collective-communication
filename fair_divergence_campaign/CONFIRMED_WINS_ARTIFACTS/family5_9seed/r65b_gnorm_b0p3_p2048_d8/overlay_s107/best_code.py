def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    dtype = x.dtype
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    s_abs = s.abs()
    g = 1.0 + BETA * s_abs.mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc_scaled = acc / W
    A = acc_scaled.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc_scaled * gr
    
    # Iteration 2
    s_abs = s.abs()
    g = 1.0 + BETA * s_abs.mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc_scaled = acc / W
    A = acc_scaled.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc_scaled * gr
    
    # Iteration 3
    s_abs = s.abs()
    g = 1.0 + BETA * s_abs.mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc_scaled = acc / W
    A = acc_scaled.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc_scaled * gr
    
    # Iteration 4
    s_abs = s.abs()
    g = 1.0 + BETA * s_abs.mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s