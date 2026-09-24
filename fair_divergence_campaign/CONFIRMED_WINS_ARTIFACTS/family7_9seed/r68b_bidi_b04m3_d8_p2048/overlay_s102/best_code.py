def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coefficients
    b = [0.4 + 0.08 * (r % 3) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply bidiagonal coupling, inverse previous coupling, and all-reduce
    for stage in range(7):
        # Apply inverse coupling from previous stage (if stage > 0)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
        
        # Divide by W and apply forward bidiagonal coupling
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s