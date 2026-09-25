def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute global coefficients
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.8 + 0.02*r
        wo = 0.15 + 0.01*(r % 5)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Pre-compute local rank parameters
    sd = rank % 5
    kd = [(sd + j) % 5 for j in range(4)]
    so = (rank + 1) % 5
    ko = set([(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1])
    wd = 0.8 + 0.02*rank
    wo = 0.15 + 0.01*(rank % 5)
    
    # Main 6 iterations with solve
    for iter in range(6):
        buf = torch.zeros_like(s)
        # Combined loop - ko is subset of kd
        for b in kd:
            val = wd * s[b*S:(b+1)*S]
            if b in ko:
                val = val + wo * s[(b-1)*S:b*S]
            buf[b*S:(b+1)*S] = val
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve triangular system in-place
        acc[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            acc[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * acc[(b-1)*S:b*S]) / A[b]
        s = acc
    
    # Final iteration without solve
    buf = torch.zeros_like(s)
    for b in kd:
        val = wd * s[b*S:(b+1)*S]
        if b in ko:
            val = val + wo * s[(b-1)*S:b*S]
        buf[b*S:(b+1)*S] = val
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s