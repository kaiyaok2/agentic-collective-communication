def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    W = world_size
    
    # Precompute counts: how many ranks contribute to each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Initial full all-reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Now perform 7 iterations of masked all-reduce using reduce-scatter + all-gather
    for iteration in range(7):
        # Determine which blocks this rank keeps
        start = (rank + OFF) % B
        keep = set((start + 1*j) % B for j in range(5))
        
        # Strategy: reduce-scatter then all-gather
        # For reduce-scatter: partition the tensor into chunks and reduce only relevant ones
        # Since keep-set is sparse (5 out of 12 blocks), we:
        # 1. Extract blocks we care about
        # 2. Perform reduce-scatter on those blocks
        # 3. All-gather to reconstruct
        
        # However, xm.reduce_scatter expects uniform chunk division across ranks
        # Instead, we use a different decomposition:
        # - Scatter phase: each rank sends its kept blocks to others
        # - Gather phase: reconstruct full tensor
        
        # Simpler approach with available primitives:
        # Use all_reduce on masked tensor (zero out non-kept blocks)
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer (this is the "reduce" part)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by contribution counts (only on last iteration we skip this)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s