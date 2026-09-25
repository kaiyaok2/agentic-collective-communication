def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # First all-reduce: sum all x across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A: the normalization factors for each block
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 5
        for j in range(3):
            A[(start + 1 * j) % 5] += w
    
    # Perform 7 weighted iterations
    for iteration in range(7):
        # Compute this rank's weight
        w = 0.5 + 0.02 * rank
        
        # Compute which blocks this rank contributes to
        start = (rank + 0) % 5
        keep = set((start + 1 * j) % 5 for j in range(3))
        
        # Mask: zero out blocks not in keep, scale blocks in keep by w
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in keep:
                buf[b * S:(b + 1) * S] = w * s[b * S:(b + 1) * S]
        
        # All-reduce to sum the masked buffers
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by A (except on the last iteration)
        if iteration < 6:
            for b in range(5):
                acc[b * S:(b + 1) * S] = acc[b * S:(b + 1) * S] / A[b]
        
        s = acc
    
    return s