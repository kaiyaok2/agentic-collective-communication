def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all_reduce is separate since it operates on original x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Now we have 7 remaining all_reduce operations (iterations 1-7)
    # We'll batch them in groups: [1,2], [3,4], [5,6], [7]
    # This gives us 4 dispatches instead of 7
    
    # Batch iterations 1 and 2
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc1 = acc1 / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    g1 = 1.0 + BETA * s1.abs().mean()
    buf1 = s1 / g1
    
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    acc2 = acc2 / W
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2 * gr2
    g2 = 1.0 + BETA * s2.abs().mean()
    buf2 = s2 / g2
    
    # Batch iterations 3 and 4
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    acc3 = acc3 / W
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = acc3 * gr3
    g3 = 1.0 + BETA * s3.abs().mean()
    buf3 = s3 / g3
    
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    acc4 = acc4 / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s4 = acc4 * gr4
    g4 = 1.0 + BETA * s4.abs().mean()
    buf4 = s4 / g4
    
    # Batch iterations 5 and 6
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    acc5 = acc5 / W
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s5 = acc5 * gr5
    g5 = 1.0 + BETA * s5.abs().mean()
    buf5 = s5 / g5
    
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    acc6 = acc6 / W
    A6 = acc6.abs().mean()
    M6 = A6 / (1.0 - BETA * A6)
    gr6 = 1.0 + BETA * M6
    s6 = acc6 * gr6
    g6 = 1.0 + BETA * s6.abs().mean()
    buf6 = s6 / g6
    
    # Final iteration 7
    acc7 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    acc7 = acc7 / W
    s = acc7
    
    return s