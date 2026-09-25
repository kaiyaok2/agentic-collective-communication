def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain strategy:
    Performs 8 sequential all-reduce operations with interleaved local 
    permutation, weighting, and normalization steps.
    """
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    dtype = x.dtype
    
    # First all-reduce: sum input x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute normalization factors A[b] for each block
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Perform 7 iterations of: weight + permute, all-reduce, normalize + inverse permute
    for iteration in range(7):
        # Determine which blocks this rank contributes to
        start = (rank + 1) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        w = 0.45 + 0.02 * (rank % 4)
        
        # Apply weighting and permutation
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b * S:(b + 1) * S] = w * s[jb * S:(jb + 1) * S]
        
        # All-reduce the weighted and permuted buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # For iterations 0-5: normalize and apply inverse permutation
        # For iteration 6 (last): skip normalization (just return acc)
        if iteration < 6:
            rec = acc.clone()
            for b in range(6):
                jb = SIG[b]
                rec[jb * S:(jb + 1) * S] = acc[b * S:(b + 1) * S] / A[b]
            s = rec
        else:
            s = acc
    
    return s