def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive Sequential Weighted All-Reduce implementation.
    Performs 1 initial all-reduce for global sum, followed by 7 dependent 
    all-reduces where each rank applies its weight to selected blocks, 
    accumulates, and normalizes.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Initial all-reduce: global sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute normalization factors A[b] for each block
    # A[b] = sum of weights over all ranks whose window covers block b
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 6
        for j in range(5):
            block_idx = (start + 1 * j) % 6
            A[block_idx] += w
    
    # Perform 7 dependent all-reduce operations
    for iteration in range(7):
        # Determine which blocks this rank should contribute to
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # This rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer: apply weight only to blocks in keep set
        buf = torch.zeros_like(s, dtype=dtype)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce to accumulate weighted contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by A[b] for each block (except last iteration)
        if iteration < 6:
            for b in range(6):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        # Update s for next iteration
        s = acc
    
    return s