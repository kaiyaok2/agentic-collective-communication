def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute coupling coefficients
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    # Strategy: Merge multiple coupling/uncoupling operations algebraically
    # Each stage does: uncouple -> couple -> all_reduce
    # We can compose these transformations to reduce all_reduce calls
    
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Buffer transformations for stages 2-8 (7 stages)
    # Each stage: uncouple previous -> couple current -> reduce -> average
    # We'll group stages to reduce all_reduce calls
    
    # Group 1: Stages 2-3 (2 stages)
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Group 2: Stages 4-5 (2 stages)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Group 3: Stages 6-7 (2 stages)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Group 4: Stage 8 (final stage)
    for r in range(W - 2, -1, -1):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    buf = s / W
    for r in range(W - 1):
        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s