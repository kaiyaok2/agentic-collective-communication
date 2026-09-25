def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version: Reduce collective dispatch count by batching operations
    and using collective_permute where beneficial.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute the normalization factors A[b] for each block
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 5
        for j in range(3):
            block_idx = (start + 1 * j) % 5
            A[block_idx] += w
    
    # Precompute which blocks this rank contributes to (constant across iterations)
    start = (rank + 0) % 5
    keep = set((start + 1 * j) % 5 for j in range(3))
    w = 0.5 + 0.02 * rank
    
    # Batch iterations 0-3 together
    for batch_start in range(0, 7, 4):
        batch_size = min(4, 7 - batch_start)
        
        # Create batched buffer for multiple iterations
        batched_buf = torch.zeros(batch_size * 5 * S, dtype=dtype, device=x.device)
        
        for i in range(batch_size):
            iteration = batch_start + i
            # Create masked weighted contribution for this iteration
            for b in range(5):
                offset = i * 5 * S + b * S
                if b in keep:
                    batched_buf[offset:offset+S] = w * s[b*S:(b+1)*S]
        
        # Single all-reduce for the batch
        batched_acc = xm.all_reduce(xm.REDUCE_SUM, batched_buf)
        
        # Process each iteration in the batch
        for i in range(batch_size):
            iteration = batch_start + i
            # Extract this iteration's result
            acc = batched_acc[i * 5 * S:(i + 1) * 5 * S]
            
            # Normalize each block (except on the last iteration)
            if iteration < 6:
                for b in range(5):
                    acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
            
            s = acc
    
    return s