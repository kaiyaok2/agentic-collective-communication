def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    W = world_size
    
    # Precompute the count for each block across all ranks
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Precompute inverse counts as a tensor for faster operations
    inv_c = torch.tensor([1.0/c[b] if c[b] > 0 else 0.0 for b in range(B)], 
                         dtype=x.dtype, device=x.device).repeat_interleave(S)
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Create a persistent mask tensor (reuse across stages)
    mask = torch.zeros(B*S, dtype=x.dtype, device=x.device)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Perform 7 dependent stages with optimized operations
    for stage in range(7):
        # Mask and all-reduce in one operation chain
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize using precomputed inverse counts (skip on last stage)
        if stage < 6:
            s = acc * inv_c
        else:
            s = acc
    
    return s