def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute coupling coefficients
    b = [0.18 + 0.13*(r % 4) for r in range(W)]
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused operation: coupling (forward sweep) for next stage
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    
    # Stages 2-8: each stage does all_reduce, then fused uncoupling (backward) + coupling (forward)
    for stage in range(2, 9):
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if stage < 8:
            # Fused: uncoupling backward sweep + coupling forward sweep
            # First uncouple (backward sweep)
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
            
            # Then couple (forward sweep) into buf
            buf = s / W
            for r in range(W - 1):
                buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    
    return s