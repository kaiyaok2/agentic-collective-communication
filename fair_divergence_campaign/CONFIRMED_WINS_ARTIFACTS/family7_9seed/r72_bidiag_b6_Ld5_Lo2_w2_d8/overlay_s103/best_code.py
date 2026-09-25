def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute A and C coefficients (same for all iterations)
    A = [0.0] * 6
    C = [0.0] * 6
    for r in range(W):
        wd = 0.7 + 0.03 * (r % 5)
        wo = 0.2 + 0.01 * (r % 7)
        sd = (r + 2) % 6
        for j in range(5):
            A[(sd + j) % 6] += wd
        so = (r + 3) % 6
        for j in range(2):
            b = (so + j) % 6
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific parameters (constant across iterations)
    sd = (rank + 2) % 6
    kd = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko = set((so + j) % 6 for j in range(2))
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Initial all-reduce (dispatch 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Prepare buffer for iteration 0 while initial all-reduce completes
    # (overlapping opportunity conceptually, but we need s first)
    
    # Main pipelined loop: 7 iterations
    for it in range(7):
        # Buffer preparation for current iteration
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in kd:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
            if b in ko and b >= 1:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # All-reduce dispatch (dispatches 2-8)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Bidiagonal solve (can overlap with next iteration's prep, but we need acc)
        if it < 6:  # For iterations 0-5, perform solve and continue
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 6):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:  # Last iteration (it == 6), no solve needed
            s = acc
    
    return s