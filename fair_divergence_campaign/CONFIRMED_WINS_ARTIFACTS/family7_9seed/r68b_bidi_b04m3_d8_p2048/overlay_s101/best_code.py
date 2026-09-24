def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute b coefficients
    b = [0.4 + 0.08 * (r % 3) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Each stage does:
    # 1. Undo previous coupling (decouple)
    # 2. Apply new coupling
    # 3. All-reduce
    for stage in range(7):
        # Decouple from previous stage (except after stage 1)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
        
        # Apply coupling for current stage
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s