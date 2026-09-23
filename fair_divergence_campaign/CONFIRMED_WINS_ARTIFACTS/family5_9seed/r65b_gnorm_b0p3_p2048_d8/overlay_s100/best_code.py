def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    dtype = x.dtype
    
    # Phase 1: Process first 4 iterations with overlapping
    # Iteration 0
    s0 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    g0 = 1.0 + BETA * s0.abs().mean()
    buf0 = s0 / g0
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Iteration 2
    acc0 = acc0 / W
    A0 = acc0.abs().mean()
    M0 = A0 / (1.0 - BETA * A0)
    gr0 = 1.0 + BETA * M0
    s1 = acc0 * gr0
    g1 = 1.0 + BETA * s1.abs().mean()
    buf1 = s1 / g1
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Iteration 3
    acc1 = acc1 / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s2 = acc1 * gr1
    g2 = 1.0 + BETA * s2.abs().mean()
    buf2 = s2 / g2
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    
    # Phase 2: Process next 4 iterations with overlapping
    # Iteration 4
    acc2 = acc2 / W
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s3 = acc2 * gr2
    g3 = 1.0 + BETA * s3.abs().mean()
    buf3 = s3 / g3
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    
    # Iteration 5
    acc3 = acc3 / W
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s4 = acc3 * gr3
    g4 = 1.0 + BETA * s4.abs().mean()
    buf4 = s4 / g4
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    
    # Iteration 6
    acc4 = acc4 / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s5 = acc4 * gr4
    g5 = 1.0 + BETA * s5.abs().mean()
    buf5 = s5 / g5
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    
    # Iteration 7 (final)
    acc5 = acc5 / W
    s = acc5
    
    return s