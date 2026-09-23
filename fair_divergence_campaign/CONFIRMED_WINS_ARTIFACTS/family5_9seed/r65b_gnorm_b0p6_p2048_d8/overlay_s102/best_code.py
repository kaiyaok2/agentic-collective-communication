def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.6
    dtype = x.dtype
    
    # Step 1: Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Issue all 6 remaining all-reduces asynchronously
    # We need to compute intermediate values that feed into all_reduces
    # But we can prepare buffers and issue operations before waiting
    
    # Compute initial normalization
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Issue first iteration all_reduce
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Continue with computations that depend on acc1
    acc1 = acc1 / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    g1 = 1.0 + BETA * s1.abs().mean()
    buf1 = s1 / g1
    
    # Issue second iteration all_reduce
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Continue with computations that depend on acc2
    acc2 = acc2 / W
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2 * gr2
    g2 = 1.0 + BETA * s2.abs().mean()
    buf2 = s2 / g2
    
    # Issue third iteration all_reduce
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    
    # Continue with computations that depend on acc3
    acc3 = acc3 / W
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = acc3 * gr3
    g3 = 1.0 + BETA * s3.abs().mean()
    buf3 = s3 / g3
    
    # Issue fourth iteration all_reduce
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    
    # Continue with computations that depend on acc4
    acc4 = acc4 / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s4 = acc4 * gr4
    g4 = 1.0 + BETA * s4.abs().mean()
    buf4 = s4 / g4
    
    # Issue fifth iteration all_reduce
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    
    # Continue with computations that depend on acc5
    acc5 = acc5 / W
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s5 = acc5 * gr5
    g5 = 1.0 + BETA * s5.abs().mean()
    buf5 = s5 / g5
    
    # Issue sixth iteration all_reduce
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    
    # Final computation
    acc6 = acc6 / W
    s_final = acc6
    
    return s_final