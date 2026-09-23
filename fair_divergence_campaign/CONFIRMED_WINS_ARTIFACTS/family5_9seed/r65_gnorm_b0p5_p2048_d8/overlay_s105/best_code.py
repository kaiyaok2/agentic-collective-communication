def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce (iteration 0)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6: Try to minimize collective dispatches
    # by batching operations where possible
    
    # Unroll and fuse operations to reduce dispatch overhead
    for i in range(6):
        # Compute normalization and buffer in one fused block
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Post-processing in one fused block
        if i < 5:  # iterations 1-5 need full processing
            acc = acc / W
            A = acc.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s = acc * gr
        else:  # iteration 6 (final)
            s = acc / W
    
    return s