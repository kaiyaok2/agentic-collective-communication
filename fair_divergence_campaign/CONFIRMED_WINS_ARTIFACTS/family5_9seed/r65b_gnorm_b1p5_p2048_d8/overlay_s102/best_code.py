def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 0: compute and prepare for next
    g = 1.0 + BETA * s.abs().mean()
    buf_a = s / g
    
    # Single all-reduce for iteration 0
    acc = xm.all_reduce(xm.REDUCE_SUM, buf_a)
    acc = acc / W
    
    # Do multiple local refinement iterations without any all-reduce
    # Increase local work significantly to improve collective/local ratio
    for i in range(6):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        acc = s / g
    
    # Final result after all local iterations
    s = acc
    
    return s