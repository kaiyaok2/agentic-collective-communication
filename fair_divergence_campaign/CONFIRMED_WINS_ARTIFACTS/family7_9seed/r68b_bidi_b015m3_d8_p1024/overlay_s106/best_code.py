def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute b coefficients
    b = [0.15 + 0.1 * (r % 3) for r in range(W)]
    
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply coupling, all_reduce, undo coupling
    for stage in range(7):
        # Apply bidiagonal coupling
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        
        # All_reduce the coupled result
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Undo the coupling (except on last stage)
        if stage < 6:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    return s