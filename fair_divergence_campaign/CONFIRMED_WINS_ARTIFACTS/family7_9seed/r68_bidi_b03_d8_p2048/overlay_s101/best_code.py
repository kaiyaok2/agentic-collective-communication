def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coupling coefficients
    b = [0.3 + 0.1 * (r % 4) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply bidiagonal coupling, undo previous, and reduce
    for stage in range(7):
        # Undo previous stage's coupling (except for first iteration after stage 1)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
        
        # Apply bidiagonal coupling: buf[r] = (s[r] + b[r]*s[r+1]) / W
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce the coupled result
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s