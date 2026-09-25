
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute global coefficients
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
    
    # Precompute rank-specific parameters
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    
    # Main iteration loop
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(5):
            contrib = None
            if b in kd:
                contrib = wd * s[b*S:(b+1)*S]
            if b in ko and b >= 1:
                off_contrib = wo * s[(b-1)*S:b*S]
                contrib = off_contrib if contrib is None else contrib + off_contrib
            if contrib is not None:
                buf[b*S:(b+1)*S] = contrib
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal system
        rec = acc.clone()
        rec[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
        s = rec
    
    # Last iteration
    buf = torch.zeros_like(s)
    for b in range(5):
        contrib = None
        if b in kd:
            contrib = wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            off_contrib = wo * s[(b-1)*S:b*S]
            contrib = off_contrib if contrib is None else contrib + off_contrib
        if contrib is not None:
            buf[b*S:(b+1)*S] = contrib
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
