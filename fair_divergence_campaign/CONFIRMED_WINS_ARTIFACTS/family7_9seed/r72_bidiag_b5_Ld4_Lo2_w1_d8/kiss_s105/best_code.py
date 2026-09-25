
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute global coefficients A and C
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.15 + 0.01 * (r % 5)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific values
    sd = rank % 5
    kd_set = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko_set = set((so + j) % 5 for j in range(2) if (so + j) % 5 >= 1)
    wd = 0.8 + 0.02 * rank
    wo = 0.15 + 0.01 * (rank % 5)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Main iteration loop - 6 full iterations
    for _ in range(6):
        # Create buffer with weighted contributions - single pass over all buckets
        buf = torch.zeros_like(s)
        for b in range(5):
            in_kd = b in kd_set
            in_ko = b in ko_set
            if in_kd and in_ko:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif in_kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif in_ko:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # All-reduce the buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve the bidiagonal system in-place
        acc[0:S] = acc[0:S] / A[0]
        acc[S:2*S] = (acc[S:2*S] - C[1] * acc[0:S]) / A[1]
        acc[2*S:3*S] = (acc[2*S:3*S] - C[2] * acc[S:2*S]) / A[2]
        acc[3*S:4*S] = (acc[3*S:4*S] - C[3] * acc[2*S:3*S]) / A[3]
        acc[4*S:5*S] = (acc[4*S:5*S] - C[4] * acc[3*S:4*S]) / A[4]
        s = acc
    
    # Final iteration
    buf = torch.zeros_like(s)
    for b in range(5):
        in_kd = b in kd_set
        in_ko = b in ko_set
        if in_kd and in_ko:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        elif in_kd:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        elif in_ko:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
