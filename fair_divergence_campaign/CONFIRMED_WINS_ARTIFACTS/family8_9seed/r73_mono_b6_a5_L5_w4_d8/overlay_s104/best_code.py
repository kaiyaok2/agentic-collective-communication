def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    dtype = x.dtype
    
    # Compute A[b] normalization factors
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused batching strategy: process iterations in batches
    # We perform 7 iterations (after initial all-reduce), batching pairs where possible
    # Batch 1: iterations 0-1 (2 all-reduces)
    # Batch 2: iterations 2-3 (2 all-reduces)
    # Batch 3: iterations 4-5 (2 all-reduces)
    # Batch 4: iteration 6 (1 all-reduce, no normalization)
    
    start_block = (rank + 1) % 6
    keep = set((start_block + 1 * j) % 6 for j in range(5))
    w = 0.45 + 0.02 * (rank % 4)
    
    # Batch iterations 0-1
    for iter_idx in range(2):
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        rec = acc.clone()
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        s = rec
    
    # Batch iterations 2-3
    for iter_idx in range(2):
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        rec = acc.clone()
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        s = rec
    
    # Batch iterations 4-5
    for iter_idx in range(2):
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        rec = acc.clone()
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        s = rec
    
    # Final iteration 6 (no normalization)
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            jb = SIG[b]
            buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s