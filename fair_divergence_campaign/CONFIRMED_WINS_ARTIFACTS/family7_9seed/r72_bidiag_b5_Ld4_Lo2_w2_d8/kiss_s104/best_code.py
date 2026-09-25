
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Pre-compute A and C coefficients once
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.7 + 0.03 * (r % 5)
        wo = 0.2 + 0.01 * (r % 7)
        sd = (r + 1) % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 2) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Pre-compute rank-specific parameters
    sd = (rank + 1) % 5
    so = (rank + 2) % 5
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Pre-compute block operations
    kd_set = set((sd + j) % 5 for j in range(4))
    ko_set = set((so + j) % 5 for j in range(2) if (so + j) % 5 >= 1)
    
    kd_only = sorted(kd_set - ko_set)
    ko_only = sorted(ko_set - kd_set)
    both = sorted(kd_set & ko_set)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(7):
        buf = torch.zeros_like(s)
        
        # Optimized buffer filling - single pass per block
        for b in kd_only:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        
        for b in ko_only:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        for b in both:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration == 6:
            return acc
        
        # Solve tridiagonal system
        rec = torch.zeros_like(acc)
        rec[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
        
        s = rec
    
    return s
