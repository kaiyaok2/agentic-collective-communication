def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Initial all-reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute normalization factors A[b] for each block
    # A[b] = sum over all ranks r of weight(r) if block b is in rank r's window
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 5
        for j in range(3):
            block_idx = (start + 1 * j) % 5
            A[block_idx] += w
    
    # Perform 7 weighted iterations
    for iteration in range(7):
        # Determine which blocks this rank contributes to
        start = (rank + 0) % 5
        keep = set((start + 1 * j) % 5 for j in range(3))
        
        # Compute this rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer: apply weight to blocks in keep set, zero elsewhere
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in keep:
                buf[b * S:(b + 1) * S] = w * s[b * S:(b + 1) * S]
        
        # All-reduce to sum weighted contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize each block by its total weight (except last iteration)
        if iteration < 6:
            for b in range(5):
                acc[b * S:(b + 1) * S] = acc[b * S:(b + 1) * S] / A[b]
        
        # Update state
        s = acc
    
    return s