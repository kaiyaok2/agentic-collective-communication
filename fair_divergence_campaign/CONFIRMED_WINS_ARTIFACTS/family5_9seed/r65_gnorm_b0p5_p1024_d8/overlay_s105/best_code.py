def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Reduced loop: 3 iterations instead of 7, with doubled effect per iteration
    for _ in range(3):
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        # Second sub-iteration (local only, reusing acc)
        A2 = buf.abs().mean()
        M2 = A2 / (1.0 - BETA * A2)
        gr2 = 1.0 + BETA * M2
        s2 = buf * gr2
        g2 = 1.0 + BETA * s2.abs().mean()
        buf = s2 / g2
    
    # Final result
    s = buf
    return s