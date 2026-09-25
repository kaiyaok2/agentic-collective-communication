def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute A and C arrays once
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
    
    # Precompute sets and weights for this rank
    sd = (rank + 2) % 6
    kd = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko = set((so + j) % 6 for j in range(2))
    wd_val = 0.7 + 0.03 * (rank % 5)
    wo_val = 0.2 + 0.01 * (rank % 7)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 iterations
    for iteration in range(7):
        # Build buffer
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in kd and b in ko and b >= 1:
                buf[b*S:(b+1)*S] = wd_val * s[b*S:(b+1)*S] + wo_val * s[(b-1)*S:b*S]
            elif b in kd:
                buf[b*S:(b+1)*S] = wd_val * s[b*S:(b+1)*S]
            elif b in ko and b >= 1:
                buf[b*S:(b+1)*S] = wo_val * s[(b-1)*S:b*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve bidiagonal system (except last iteration)
        if iteration < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 6):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:
            s = acc
    
    return s