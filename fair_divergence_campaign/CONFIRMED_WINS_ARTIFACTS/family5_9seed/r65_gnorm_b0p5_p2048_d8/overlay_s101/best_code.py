def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initial all-reduce for s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batch iterations 1-3 together
    # Iteration 1: prepare buffer
    g1 = 1.0 + BETA * s.abs().mean()
    buf1 = s / g1
    
    # Iteration 2: prepare buffer (using s for now, will update after all-reduce)
    # We need to compute this after getting acc1, so we'll do it in batches
    
    # Do iteration 1 all-reduce
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1) / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    
    # Prepare iteration 2
    g2 = 1.0 + BETA * s1.abs().mean()
    buf2 = s1 / g2
    
    # Do iteration 2 all-reduce
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2) / W
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2 * gr2
    
    # Prepare iterations 3 and 4
    g3 = 1.0 + BETA * s2.abs().mean()
    buf3 = s2 / g3
    
    # Batch all-reduce for iterations 3 and 4
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3) / W
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = acc3 * gr3
    
    g4 = 1.0 + BETA * s3.abs().mean()
    buf4 = s3 / g4
    
    # Batch iterations 4 and 5
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4) / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s4 = acc4 * gr4
    
    g5 = 1.0 + BETA * s4.abs().mean()
    buf5 = s4 / g5
    
    # Batch iterations 5 and 6
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5) / W
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s5 = acc5 * gr5
    
    g6 = 1.0 + BETA * s5.abs().mean()
    buf6 = s5 / g6
    
    # Final all-reduce
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    
    return acc6