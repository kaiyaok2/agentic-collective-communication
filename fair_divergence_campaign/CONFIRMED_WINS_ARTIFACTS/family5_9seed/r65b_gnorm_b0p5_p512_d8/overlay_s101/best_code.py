def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initial all_reduce for x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # We have 6 iterations after the initial all_reduce
    # Each iteration: compute g, normalize, all_reduce, compute A, M, gr
    # Strategy: Bundle scalar metadata (g, A, M, gr) for multiple iterations
    # and synchronize them together to reduce dispatches
    
    # Iteration 1
    g1 = 1.0 + BETA * s.abs().mean()
    buf1 = s / g1
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    acc1 = acc1 / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    
    # Iteration 2
    g2 = 1.0 + BETA * s1.abs().mean()
    buf2 = s1 / g2
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    acc2 = acc2 / W
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2 * gr2
    
    # Iteration 3
    g3 = 1.0 + BETA * s2.abs().mean()
    buf3 = s2 / g3
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    acc3 = acc3 / W
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = acc3 * gr3
    
    # Iteration 4
    g4 = 1.0 + BETA * s3.abs().mean()
    buf4 = s3 / g4
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    acc4 = acc4 / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s4 = acc4 * gr4
    
    # Iteration 5
    g5 = 1.0 + BETA * s4.abs().mean()
    buf5 = s4 / g5
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    acc5 = acc5 / W
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s5 = acc5 * gr5
    
    # Iteration 6 (final)
    g6 = 1.0 + BETA * s5.abs().mean()
    buf6 = s5 / g6
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    acc6 = acc6 / W
    
    return acc6