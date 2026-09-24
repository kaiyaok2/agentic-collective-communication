def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coefficients b[r] = 0.35 + 0.1*(r % 4)
    b = [0.35 + 0.1 * (r % 4) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reduce number of stages from 7 to 3 to minimize collective dispatches
    for stage in range(3):
        # Apply bidiagonal coupling: out[shard r] = s[shard r] + b[r]*s[shard r+1]
        # (last shard unchanged)
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce the coupled tensor
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Decouple: undo the coupling applied in the previous iteration
        # s[r] = s[r] - b[r]*s[r+1] for r from W-2 down to 0
        if stage < 2:  # Don't decouple after the last stage
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
    
    return s