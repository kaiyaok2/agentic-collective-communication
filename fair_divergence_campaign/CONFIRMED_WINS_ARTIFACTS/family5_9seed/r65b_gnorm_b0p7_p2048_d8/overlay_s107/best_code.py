def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    dtype = x.dtype
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iterations with 2x unrolling (reduce to 3 more all-reduces)
    for _ in range(3):
        # First sub-iteration
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Second sub-iteration (local only, no all-reduce)
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Final all-reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s