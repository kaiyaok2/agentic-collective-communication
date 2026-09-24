def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coefficients
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Each stage applies bidi-coupling, reduces, then decouples
    # Strategy: Fuse coupling into buffer prep to overlap with communication completion
    for stage in range(7):
        # Decouple from previous stage (skip for stage 0 after initial reduce)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        
        # Prepare buffer with bidi-coupling fused into averaging
        # This overlaps local compute with any residual collective latency
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce the coupled buffer
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final result: s already contains the 8th stage output
    # No final decoupling needed as per problem spec (final result is coupled)
    return s