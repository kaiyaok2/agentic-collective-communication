
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C arrays
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
    
    # Precompute rank-specific parameters
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2) if (so + j) % 5 >= 1)
    wd = 0.8 + 0.02 * rank
    wo = 0.15 + 0.01 * (rank % 5)
    
    # Separate blocks into categories for efficient processing
    both = list(kd & ko)
    kd_only = list(kd - ko)
    ko_only = list(ko - kd)
    
    # Main iteration loop
    for iteration in range(7):
        # Build buffer more efficiently
        buf = torch.zeros_like(s)
        
        # Blocks in both kd and ko - do in one operation
        for b in both:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # Blocks only in kd
        for b in kd_only:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        
        # Blocks only in ko
        for b in ko_only:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve (except on last iteration)
        if iteration < 6:
            b0 = acc[0:S] / A[0]
            b1 = (acc[S:2*S] - C[1] * b0) / A[1]
            b2 = (acc[2*S:3*S] - C[2] * b1) / A[2]
            b3 = (acc[3*S:4*S] - C[3] * b2) / A[3]
            b4 = (acc[4*S:5*S] - C[4] * b3) / A[4]
            s = torch.cat([b0, b1, b2, b3, b4], dim=0)
        else:
            s = acc
    
    return s
