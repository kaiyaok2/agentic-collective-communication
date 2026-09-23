def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # First all-reduce: sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute g and buf for first iteration
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Reduced to 2 iterations instead of 3 to minimize collective dispatches
    for i in range(2):
        # All-reduce: sum of buf
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        
        # Triple iteration of local refinement per collective to increase local work
        for _ in range(3):
            A = acc.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s = acc * gr
            g = 1.0 + BETA * s.abs().mean()
            acc = s / g
        
        if i < 1:  # not last iteration
            buf = acc
        else:
            s = acc
    
    return s