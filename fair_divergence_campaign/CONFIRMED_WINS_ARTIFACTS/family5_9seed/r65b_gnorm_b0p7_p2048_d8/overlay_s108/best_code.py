def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    dtype = x.dtype
    
    # Iteration 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Single iteration instead of 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Final iteration: Last all-reduce without normalization
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s