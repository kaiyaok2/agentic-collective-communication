
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
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
    
    # Precompute this rank's contribution pattern
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 iterations
    for iteration in range(7):
        # Build contribution buffer - optimize to avoid read-modify-write
        buf = torch.zeros_like(s)
        for b in range(5):
            has_kd = b in kd
            has_ko = b in ko and b >= 1
            if has_kd and has_ko:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif has_kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif has_ko:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # All reduce to aggregate contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Forward substitution (skip on last iteration)
        if iteration < 6:
            # Avoid clone by building result from slices
            rec_0 = acc[0:S] / A[0]
            rec_1 = (acc[S:2*S] - C[1] * rec_0) / A[1]
            rec_2 = (acc[2*S:3*S] - C[2] * rec_1) / A[2]
            rec_3 = (acc[3*S:4*S] - C[3] * rec_2) / A[3]
            rec_4 = (acc[4*S:5*S] - C[4] * rec_3) / A[4]
            s = torch.cat([rec_0, rec_1, rec_2, rec_3, rec_4], dim=0)
        else:
            s = acc
    
    return s
