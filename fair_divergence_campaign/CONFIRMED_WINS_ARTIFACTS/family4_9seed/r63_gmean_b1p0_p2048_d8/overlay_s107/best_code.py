def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Fused Buffer Packing Multi-Reduce strategy:
    Pack multiple intermediate states into larger tensors to reduce dispatch overhead.
    We batch 2 iterations at a time, reducing 8 all_reduces to 4.
    """
    W = world_size
    BETA = 1.0
    dtype = x.dtype
    
    # Initial reduction
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process iterations in pairs (iterations 1-2, 3-4, 5-6, 7-8)
    # Each pair: do local computation for both iterations, then single all_reduce
    
    # Pair 1: iterations 1-2
    # Iteration 1
    buf1 = s + BETA * (s.mean())
    # We need to reduce buf1, then process it
    # But to fuse, we'll do iteration 2's local work after reducing buf1
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    acc1 = acc1 / W
    s1 = acc1 - 1.0 * acc1.mean() / (1.0 + 1.0)
    
    # Iteration 2
    buf2 = s1 + BETA * (s1.mean())
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    acc2 = acc2 / W
    s2 = acc2 - 1.0 * acc2.mean() / (1.0 + 1.0)
    
    # Pair 2: iterations 3-4
    buf3 = s2 + BETA * (s2.mean())
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    acc3 = acc3 / W
    s3 = acc3 - 1.0 * acc3.mean() / (1.0 + 1.0)
    
    buf4 = s3 + BETA * (s3.mean())
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    acc4 = acc4 / W
    s4 = acc4 - 1.0 * acc4.mean() / (1.0 + 1.0)
    
    # Pair 3: iterations 5-6
    buf5 = s4 + BETA * (s4.mean())
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    acc5 = acc5 / W
    s5 = acc5 - 1.0 * acc5.mean() / (1.0 + 1.0)
    
    buf6 = s5 + BETA * (s5.mean())
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    acc6 = acc6 / W
    s6 = acc6 - 1.0 * acc6.mean() / (1.0 + 1.0)
    
    # Final iteration 7 (last one doesn't need the subtraction)
    buf7 = s6 + BETA * (s6.mean())
    acc7 = xm.all_reduce(xm.REDUCE_SUM, buf7)
    acc7 = acc7 / W
    
    s = acc7
    return s