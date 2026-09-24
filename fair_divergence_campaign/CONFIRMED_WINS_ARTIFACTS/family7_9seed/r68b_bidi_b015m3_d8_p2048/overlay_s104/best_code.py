def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Compute b coefficients for bidiagonal coupling
    b = [0.15 + 0.1 * (r % 3) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply bidiagonal coupling for stage 1
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    
    # Stages 2-8: Decouple previous, reduce, couple current
    for stage in range(2, 9):
        # All-reduce the coupled buffer
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Decouple (undo the previous stage's coupling)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        
        # Apply bidiagonal coupling for current stage (except last stage)
        if stage < 8:
            buf = s / W
            for r in range(W - 1):
                buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    
    # Stage 8 final result
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s