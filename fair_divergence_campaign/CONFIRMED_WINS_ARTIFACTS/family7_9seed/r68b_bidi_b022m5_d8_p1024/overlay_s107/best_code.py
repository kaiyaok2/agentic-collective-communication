def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Compute bidiagonal coefficients
    b = [0.22 + 0.11 * (r % 5) for r in range(W)]
    
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply bidiagonal coupling, inverse previous coupling, and all_reduce
    for stage in range(7):
        # Undo previous bidiagonal coupling (inverse operation)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
        
        # Apply bidiagonal coupling and divide by W
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All_reduce the coupled buffer
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s