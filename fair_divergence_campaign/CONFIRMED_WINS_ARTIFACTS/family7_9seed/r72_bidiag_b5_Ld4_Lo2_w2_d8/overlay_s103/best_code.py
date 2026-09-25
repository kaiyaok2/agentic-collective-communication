def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Pre-compute global A and C coefficients
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.7 + 0.03 * (r % 5)
        wo = 0.2 + 0.01 * (r % 7)
        sd = (r + 1) % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 2) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Pre-compute all 7 weighted buffer applications for this rank
    # We need to compute the buffer for each of the 7 iterations that depend on rank's data
    # The pattern: iteration i needs buffer based on input s from iteration i-1
    # We'll compute the weight pattern once and store all 7 weighted versions
    
    # Compute this rank's weight and window parameters
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    sd = (rank + 1) % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 2) % 5
    ko = set((so + j) % 5 for j in range(2))
    
    # Initial all-reduce to get s_0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 dependent iterations (8 all-reduces total: 1 initial + 7 iterations)
    for iteration in range(7):
        # Build buffer based on current s
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in kd:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
            if b in ko and b >= 1:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # All-reduce the buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve the lower-bidiagonal system (if this is not the last iteration)
        if iteration < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 5):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:
            # Last iteration: just return acc without solving
            s = acc
    
    return s