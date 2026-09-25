
def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*6
    C = [0.0]*6
    for r in range(W):
        wd = 0.7 + 0.03*(r % 5)
        wo = 0.2 + 0.01*(r % 7)
        sd = (r + 2) % 6
        for j in range(5):
            A[(sd + j) % 6] += wd
        so = (r + 3) % 6
        for j in range(2):
            b = (so + j) % 6
            if b >= 1:
                C[b] += wo
    sd = (rank + 2) % 6
    kd = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko = set((so + j) % 6 for j in range(2))
    wd = 0.7 + 0.03*(rank % 5)
    wo = 0.2 + 0.01*(rank % 7)
    
    # Precompute operations for each bucket
    ops = []
    for b in range(6):
        ops.append((b in kd, b in ko and b >= 1))
    
    for _ in range(5):
        buf = torch.zeros_like(s)
        for b in range(6):
            has_kd, has_ko = ops[b]
            if has_kd and has_ko:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif has_kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif has_ko:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        slices = []
        slices.append(acc[0:S] / A[0])
        for b in range(1, 6):
            slices.append((acc[b*S:(b+1)*S] - C[b] * slices[b-1]) / A[b])
        s = torch.cat(slices, dim=0)
    
    # Final iteration
    buf = torch.zeros_like(s)
    for b in range(6):
        has_kd, has_ko = ops[b]
        if has_kd and has_ko:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        elif has_kd:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        elif has_ko:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
