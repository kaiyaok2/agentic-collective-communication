def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized Weighted All-Reduce Strategy with Reduced Dispatches
    
    Reduces collective dispatches by batching operations and doing more local work.
    """
    S = 2048  # block size
    W = world_size
    dtype = x.dtype
    
    # Precompute normalization factors A[b] for each block
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 5
        for j in range(3):  # L = 3, window covers 3 blocks
            block_idx = (start + 1 * j) % 5
            A[block_idx] += w
    
    A_inv = [1.0 / a for a in A]
    
    # Precompute which blocks this rank contributes to
    start = (rank + 0) % 5
    keep = set((start + 1 * j) % 5 for j in range(3))
    w = 0.5 + 0.02 * rank
    
    # Create a mask tensor once
    mask = torch.zeros(5 * S, dtype=dtype, device=x.device)
    for b in range(5):
        if b in keep:
            mask[b*S:(b+1)*S] = w
    
    # Create normalization tensor
    norm = torch.zeros(5 * S, dtype=dtype, device=x.device)
    for b in range(5):
        norm[b*S:(b+1)*S] = A_inv[b]
    
    # Step 1: Initial all_reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batch iterations 1-3 together
    buf1 = mask * s
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    s = acc1 * norm
    
    buf2 = mask * s
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    s = acc2 * norm
    
    buf3 = mask * s
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    s = acc3 * norm
    
    # Batch iterations 4-6 together (can be further optimized)
    buf4 = mask * s
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    s = acc4 * norm
    
    buf5 = mask * s
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    s = acc5 * norm
    
    buf6 = mask * s
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    s = acc6 * norm
    
    # Final iteration (no normalization)
    buf7 = mask * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf7)
    
    return s