
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute coefficients once
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.15 + 0.01 * (r % 5)
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
    kd_list = [(sd + j) % 5 for j in range(4)]
    so = (rank + 1) % 5
    ko_list = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    wd = 0.8 + 0.02 * rank
    wo = 0.15 + 0.01 * (rank % 5)
    
    # Main iteration loop
    for iteration in range(7):
        # Pre-multiply by weights
        s_wd = wd * s
        s_wo = wo * s
        
        # Apply weights
        buf = torch.zeros_like(s)
        for b in kd_list:
            buf[b*S:(b+1)*S] = s_wd[b*S:(b+1)*S]
        for b in ko_list:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + s_wo[(b-1)*S:b*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Forward substitution (skip on last iteration)
        if iteration < 6:
            parts = [acc[0:S] / A[0]]
            for b in range(1, 5):
                parts.append((acc[b*S:(b+1)*S] - C[b] * parts[b-1]) / A[b])
            s = torch.cat(parts, dim=0)
        else:
            s = acc
    
    return s
