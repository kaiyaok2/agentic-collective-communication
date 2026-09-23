def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute coupling coefficients
    b = [0.2 + 0.1 * (r % 3) for r in range(W)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 8 stages (first stage already done, 7 more to go)
    for stage in range(7):
        # Undo previous coupling (from stage > 0)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
        
        # Apply bidiagonal coupling and average
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All_reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s