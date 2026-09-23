def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform only 2 iterations instead of 5 (significantly reducing collective count)
    for i in range(2):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Final iteration (no gr multiplication at the end)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s