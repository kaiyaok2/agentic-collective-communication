
def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute A and C arrays (constant across all iterations)
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
    
    # Pre-compute rank-specific constants
    sd = (rank + 2) % 6
    kd_set = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko_set = set((so + j) % 6 for j in range(2) if (so + j) % 6 >= 1)
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Pre-compute which blocks need which operations
    blocks_d_only = []
    blocks_o_only = []
    blocks_both = []
    for b in range(6):
        in_kd = b in kd_set
        in_ko = b in ko_set
        if in_kd and in_ko:
            blocks_both.append(b)
        elif in_kd:
            blocks_d_only.append(b)
        elif in_ko:
            blocks_o_only.append(b)
    
    # Pre-compute inverse of A
    A_inv = [1.0 / a for a in A]
    
    # Perform 6 iterations with forward substitution
    for _ in range(6):
        # Apply local transformations
        buf = torch.zeros_like(s)
        for b in blocks_d_only:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        for b in blocks_o_only:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        for b in blocks_both:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve the linear system
        acc[0:S] = acc[0:S] * A_inv[0]
        for b in range(1, 6):
            acc[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * acc[(b-1)*S:b*S]) * A_inv[b]
        
        s = acc
    
    # Final iteration without forward substitution
    buf = torch.zeros_like(s)
    for b in blocks_d_only:
        buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
    for b in blocks_o_only:
        buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
    for b in blocks_both:
        buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
