def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Determine which blocks THIS rank owns (its window)
    start = (rank + OFF) % B
    my_blocks = set((start + j) % B for j in range(L))
    
    # Precompute mask and normalization as a single tensor operation
    mask = torch.zeros_like(x)
    inv_counts = torch.zeros_like(x)
    for b in range(B):
        if b in my_blocks:
            mask[b*S:(b+1)*S] = 1.0
            inv_counts[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Stage 0: Initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 1-4: Apply mask/normalize and reduce in a tight loop
    # This reduces overhead by minimizing Python interpreter overhead
    s = s * mask * inv_counts
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    s = s * mask * inv_counts
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    s = s * mask * inv_counts
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    s = s * mask * inv_counts
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    # Final stage: mask only (no normalization)
    s = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s