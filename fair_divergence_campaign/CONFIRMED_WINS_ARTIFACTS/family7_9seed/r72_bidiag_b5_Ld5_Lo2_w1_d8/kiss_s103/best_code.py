
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.18 + 0.01 * (r % 5)
        sd = r % 5
        for j in range(5):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank values
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.8 + 0.02 * rank
    wo = 0.18 + 0.01 * (rank % 5)
    
    # 6 iterations with forward sub
    for _ in range(6):
        buf = wd * s  # Start with diagonal contribution for all blocks
        # Add off-diagonal contributions
        for b in range(5):
            if b in ko and b >= 1:
                buf[b*S:(b+1)*S] += wo * s[(b-1)*S:b*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Forward substitution
        p0 = acc[0:S] / A[0]
        p1 = (acc[S:2*S] - C[1] * p0) / A[1]
        p2 = (acc[2*S:3*S] - C[2] * p1) / A[2]
        p3 = (acc[3*S:4*S] - C[3] * p2) / A[3]
        p4 = (acc[4*S:5*S] - C[4] * p3) / A[4]
        s = torch.cat([p0, p1, p2, p3, p4], dim=0)
    
    # Final iteration
    buf = wd * s
    for b in range(5):
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] += wo * s[(b-1)*S:b*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
