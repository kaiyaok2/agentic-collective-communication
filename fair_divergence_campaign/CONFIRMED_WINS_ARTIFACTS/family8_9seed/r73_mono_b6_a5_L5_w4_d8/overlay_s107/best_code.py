def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    dtype = x.dtype
    
    # Precompute A[b] - the sum of weights for each block
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Initial all-reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process 7 iterations (first iteration is the initial all-reduce, then 6 more)
    for iteration in range(6):
        # Determine which blocks this rank should keep
        start = (rank + 1) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # Get weight for this rank
        w = 0.45 + 0.02 * (rank % 4)
        
        # Create buffer with weighted permuted values only for kept blocks
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        
        # All-reduce the buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unpermute and normalize (except on last iteration)
        if iteration < 5:
            rec = acc.clone()
            for b in range(6):
                jb = SIG[b]
                rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
            s = rec
        else:
            # Last iteration - just return accumulated result
            s = acc
    
    return s