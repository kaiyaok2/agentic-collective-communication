
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Compute global coefficients A and C
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.9 + 0.04 * (r % 3)
        wo = 0.25 + 0.01 * (r % 3)
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
    kd = {(sd + j) % 5 for j in range(4)}
    so = (rank + 1) % 5
    ko = {(so + j) % 5 for j in range(2)}
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Main iterations
    for iter_idx in range(7):
        # Build buffer with rank-specific weights - avoid += by computing sum directly
        buf = torch.zeros_like(s)
        for b in range(5):
            in_kd = b in kd
            in_ko = b in ko and b >= 1
            if in_kd and in_ko:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif in_kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif in_ko:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve bidiagonal system (skip on last iteration)
        if iter_idx < 6:
            acc[0:S] = acc[0:S] / A[0]
            for b in range(1, 5):
                acc[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * acc[(b-1)*S:b*S]) / A[b]
        
        s = acc
    
    return s
