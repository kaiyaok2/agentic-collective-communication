def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.35 + 0.1*(r % 4) for r in range(W)]
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pipelined two-stage with overlap: process stages in pairs
    # Stage pairs: (1,2), (3,4), (5,6), (7,8) - but stage 1 is just the initial reduce
    # So we have pairs: (2,3), (4,5), (6,7), and final stage 8
    
    # Actually, let's interpret the strategy as: overlap computation for next stage
    # while all-reduce is in flight by using double buffering
    
    # We'll use two buffers and alternate between them
    buf_a = s / W
    buf_b = torch.zeros_like(s)
    
    # Stage 2
    for r in range(W - 1):
        buf_a[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf_a)
    
    # Stages 3-8: pipeline by preparing next coupling while reduce is conceptually in-flight
    # In practice, xm.all_reduce is blocking, but we structure code to be ready for async
    
    for stage in range(2, 8):
        # Undo previous coupling
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        
        # Apply new coupling (using alternate buffer conceptually)
        if stage % 2 == 0:
            buf_curr = buf_a
        else:
            buf_curr = buf_b
            
        buf_curr[:] = s / W
        for r in range(W - 1):
            buf_curr[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf_curr)
    
    return s