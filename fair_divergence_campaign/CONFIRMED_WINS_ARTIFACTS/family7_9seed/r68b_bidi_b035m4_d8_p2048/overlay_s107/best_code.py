def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.35 + 0.1*(r % 4) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply bidiagonal coupling, reduce, then inverse coupling
    for stage in range(7):
        # Apply bidiagonal coupling: out[shard r] = s[shard r] + b[r]*s[shard r+1]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Inverse coupling to undo the coupling for next iteration (except last)
        if stage < 6:
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
    
    return s