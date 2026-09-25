
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C arrays
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
    
    # Precompute rank-specific constants
    sd = (rank + 1) % 5
    kd = [(sd + j) % 5 for j in range(4)]  # List instead of set
    so = (rank + 2) % 5
    ko = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]  # List instead of set
    wd = 0.7 + 0.03*(rank % 5)
    wo = 0.2 + 0.01*(rank % 7)
    
    # Find overlap between kd and ko
    kd_set = set(kd)
    ko_set = set(ko)
    overlap = kd_set & ko_set
    only_kd = kd_set - ko_set
    only_ko = ko_set - kd_set
    
    # 5 iterations with tridiagonal solve
    for _ in range(5):
        buf = torch.zeros_like(s)
        
        # Blocks only in kd
        for b in only_kd:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        
        # Blocks only in ko
        for b in only_ko:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # Blocks in both kd and ko
        for b in overlap:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Tridiagonal solve
        acc[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            acc[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * acc[(b-1)*S:b*S]) / A[b]
        
        s = acc
    
    # Final iteration without tridiagonal solve
    buf = torch.zeros_like(s)
    
    for b in only_kd:
        buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
    
    for b in only_ko:
        buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
    
    for b in overlap:
        buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
