def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute b coefficients
    b = [0.35 + 0.09 * (r % 4) for r in range(W)]
    b_rank = b[rank]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x.clone())
    
    # Stages 2-8: Apply bidiagonal coupling, undo previous, reduce
    for stage in range(7):
        # Undo previous coupling (backward pass)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        
        # Apply new coupling (forward pass) with averaging
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce the coupled result
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s