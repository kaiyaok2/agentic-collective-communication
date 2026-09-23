def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.6
    dtype = x.dtype
    
    # Combine iterations to reduce collective count
    # Instead of 7 separate all_reduces, we'll use fewer by doing more local work
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Do 2 iterations locally before next all_reduce
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All_reduce for iteration 1-2
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc1 = acc1 / W
    
    # Local compute for iterations 1-2
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    g1 = 1.0 + BETA * s1.abs().mean()
    buf1 = s1 / g1
    
    A1b = buf1.abs().mean()
    M1b = A1b / (1.0 - BETA * A1b)
    gr1b = 1.0 + BETA * M1b
    s1b = buf1 * gr1b
    g1b = 1.0 + BETA * s1b.abs().mean()
    buf1b = s1b / g1b
    
    # All_reduce for iteration 3-4
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf1b)
    acc2 = acc2 / W
    
    # Local compute for iterations 3-4
    A2 = acc2.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2 * gr2
    g2 = 1.0 + BETA * s2.abs().mean()
    buf2 = s2 / g2
    
    A2b = buf2.abs().mean()
    M2b = A2b / (1.0 - BETA * A2b)
    gr2b = 1.0 + BETA * M2b
    s2b = buf2 * gr2b
    g2b = 1.0 + BETA * s2b.abs().mean()
    buf2b = s2b / g2b
    
    # All_reduce for iteration 5-6
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf2b)
    acc3 = acc3 / W
    
    # Local compute for iterations 5-6
    A3 = acc3.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = acc3 * gr3
    g3 = 1.0 + BETA * s3.abs().mean()
    buf3 = s3 / g3
    
    # Final all_reduce
    acc_final = xm.all_reduce(xm.REDUCE_SUM, buf3)
    acc_final = acc_final / W
    
    return acc_final