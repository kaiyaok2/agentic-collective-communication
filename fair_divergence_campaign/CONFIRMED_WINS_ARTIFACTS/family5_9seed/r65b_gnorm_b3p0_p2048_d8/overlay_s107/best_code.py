def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    dtype = x.dtype
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute g, divide, and prepare for next all-reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Iterations 2-7: Each iteration does all-reduce followed by fused local ops
    for i in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Fused: divide by W, compute A, M, gr, s, g, buf
        acc = acc / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Final all-reduce (8th)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    # Final fused ops: just divide by W
    s = acc / W
    
    return s