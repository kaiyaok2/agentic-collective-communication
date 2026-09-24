def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized 8-stage sequential all-reduce with bidiagonal coupling.
    Reduced collective dispatch by combining pairs of stages.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coefficients
    b = [0.35 + 0.1 * (r % 4) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-3: Combined
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    # Apply second coupling locally
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    # Decouple twice
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Stages 4-5: Combined
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Stages 6-7: Combined
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    buf2 = buf.clone()
    for r in range(W - 1):
        buf2[r*S:(r+1)*S] = buf[r*S:(r+1)*S] + b[r]*buf[(r+1)*S:(r+2)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf2)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    # Stage 8: Final
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s