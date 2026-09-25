
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute A and C once
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
    
    # Compute rank-specific values once
    sd = (rank + 1) % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 2) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Perform 6 full iterations with backward substitution
    for _ in range(6):
        buf = torch.zeros_like(s)
        # Optimize: directly assign without reading from buf first
        for b in range(5):
            val = None
            if b in kd:
                val = wd * s[b*S:(b+1)*S]
            if b in ko and b >= 1:
                contrib = wo * s[(b-1)*S:b*S]
                val = contrib if val is None else val + contrib
            if val is not None:
                buf[b*S:(b+1)*S] = val
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward substitution
        rec0 = acc[0:S] / A[0]
        rec1 = (acc[S:2*S] - C[1] * rec0) / A[1]
        rec2 = (acc[2*S:3*S] - C[2] * rec1) / A[2]
        rec3 = (acc[3*S:4*S] - C[3] * rec2) / A[3]
        rec4 = (acc[4*S:5*S] - C[4] * rec3) / A[4]
        s = torch.cat([rec0, rec1, rec2, rec3, rec4], dim=0)
    
    # Final iteration
    buf = torch.zeros_like(s)
    for b in range(5):
        val = None
        if b in kd:
            val = wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            contrib = wo * s[(b-1)*S:b*S]
            val = contrib if val is None else val + contrib
        if val is not None:
            buf[b*S:(b+1)*S] = val
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
