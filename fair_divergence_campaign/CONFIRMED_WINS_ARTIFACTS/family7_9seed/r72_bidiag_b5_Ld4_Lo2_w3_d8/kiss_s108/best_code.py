
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute coefficients once
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.9 + 0.04*(r % 3)
        wo = 0.25 + 0.01*(r % 3)
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
    so = (rank + 1) % 5
    wd = 0.9 + 0.04*(rank % 3)
    wo = 0.25 + 0.01*(rank % 3)
    
    # Precompute which blocks get which contributions
    block_ops = []
    for b in range(5):
        in_kd = (b - sd) % 5 < 4
        in_ko = (b - so) % 5 < 2 and b >= 1
        block_ops.append((in_kd, in_ko))
    
    # 7 iterations
    for iter_num in range(7):
        buf = torch.zeros_like(s)
        for b in range(5):
            in_kd, in_ko = block_ops[b]
            if in_kd and in_ko:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif in_kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif in_ko:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Only solve for first 6 iterations
        if iter_num < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 5):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:
            s = acc
    
    return s
