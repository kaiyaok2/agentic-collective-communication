def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.3 + 0.1*(r % 5) for r in range(W)]
    
    # Initial all-reduce (stage 0)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 1-7: Use in-place operations to minimize buffer allocation
    # Each pair of stages: forward coupling, all-reduce, backward decoupling
    
    for stage_pair in range(4):  # 4 pairs covering stages 1-8, but we only do 7 stages
        if stage_pair == 3:  # Last iteration is only stage 7
            # Stage 7 only
            buf = s / W
            for r in range(W - 1):
                buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
            s = xm.all_reduce(xm.REDUCE_SUM, buf)
        else:
            # Process two stages together but more efficiently
            # Stage 2k+1
            s_scaled = s / W
            for r in range(W - 1):
                s_scaled[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
            s_reduced = xm.all_reduce(xm.REDUCE_SUM, s_scaled)
            
            # Undo coupling in-place
            for r in range(W - 2, -1, -1):
                s_reduced[r*S:(r+1)*S] = s_reduced[r*S:(r+1)*S] - b[r]*s_reduced[(r+1)*S:(r+2)*S]
            
            # Stage 2k+2
            s_scaled2 = s_reduced / W
            for r in range(W - 1):
                s_scaled2[r*S:(r+1)*S] = (s_reduced[r*S:(r+1)*S] + b[r]*s_reduced[(r+1)*S:(r+2)*S]) / W
            s_reduced2 = xm.all_reduce(xm.REDUCE_SUM, s_scaled2)
            
            # Undo coupling in-place
            for r in range(W - 2, -1, -1):
                s_reduced2[r*S:(r+1)*S] = s_reduced2[r*S:(r+1)*S] - b[r]*s_reduced2[(r+1)*S:(r+2)*S]
            
            s = s_reduced2
    
    return s