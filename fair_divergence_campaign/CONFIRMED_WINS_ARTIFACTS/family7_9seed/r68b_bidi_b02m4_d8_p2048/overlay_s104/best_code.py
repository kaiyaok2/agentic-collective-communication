def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute coupling coefficients
    b = [0.2 + 0.12 * (r % 4) for r in range(W)]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse multiple stages into fewer all-reduces
    # Process stages 1-4 with 2 all-reduces instead of 4
    
    # Stage 1 and 2 fused
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse coupling
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Forward coupling again
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Stages 3-4 fused with more local computation
    for _ in range(2):
        # Inverse coupling
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        
        # Forward coupling
        for r in range(W - 1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]
    
    # Final all-reduce after local computation
    buf = s / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s