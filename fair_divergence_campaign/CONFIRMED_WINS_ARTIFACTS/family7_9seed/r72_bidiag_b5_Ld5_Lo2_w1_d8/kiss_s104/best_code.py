
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Compute A and C once
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
    
    # Precompute reciprocals
    A_inv = [1.0 / a for a in A]
    
    # Compute rank-specific values once
    wd = 0.8 + 0.02 * rank
    wo = 0.18 + 0.01 * (rank % 5)
    r_mod = rank % 5
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 iterations with solve
    for _ in range(5):
        buf = wd * s
        # Explicit handling of rank cases
        if r_mod == 0:
            buf[S:3*S] += wo * s[0:2*S]
        elif r_mod == 1:
            buf[2*S:4*S] += wo * s[S:3*S]
        elif r_mod == 2:
            buf[3*S:5*S] += wo * s[2*S:4*S]
        elif r_mod == 3:
            buf[4*S:5*S] += wo * s[3*S:4*S]
        elif r_mod == 4:
            buf[S:2*S] += wo * s[0:S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal system
        s0 = acc[0:S] * A_inv[0]
        s1 = (acc[S:2*S] - C[1] * s0) * A_inv[1]
        s2 = (acc[2*S:3*S] - C[2] * s1) * A_inv[2]
        s3 = (acc[3*S:4*S] - C[3] * s2) * A_inv[3]
        s4 = (acc[4*S:5*S] - C[4] * s3) * A_inv[4]
        s = torch.cat([s0, s1, s2, s3, s4], dim=0)
    
    # Last iteration without solve
    buf = wd * s
    if r_mod == 0:
        buf[S:3*S] += wo * s[0:2*S]
    elif r_mod == 1:
        buf[2*S:4*S] += wo * s[S:3*S]
    elif r_mod == 2:
        buf[3*S:5*S] += wo * s[2*S:4*S]
    elif r_mod == 3:
        buf[4*S:5*S] += wo * s[3*S:4*S]
    elif r_mod == 4:
        buf[S:2*S] += wo * s[0:S]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
