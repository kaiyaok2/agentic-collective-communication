
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.7 + 0.03*(r % 5)
        wo = 0.2 + 0.01*(r % 7)
        sd = (r + 1) % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 2) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    sd = (rank + 1) % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 2) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.7 + 0.03*(rank % 5)
    wo = 0.2 + 0.01*(rank % 7)
    
    # Loop over 6 iterations
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in kd and b in ko and b >= 1:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
            elif b in kd:
                buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
            elif b in ko and b >= 1:
                buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Optimize solve step - avoid clone by building with cat
        if iteration < 5:
            parts = [acc[0:S] / A[0]]
            for b in range(1, 5):
                parts.append((acc[b*S:(b+1)*S] - C[b] * parts[b-1]) / A[b])
            s = torch.cat(parts, dim=0)
        else:
            s = acc
    
    return s
