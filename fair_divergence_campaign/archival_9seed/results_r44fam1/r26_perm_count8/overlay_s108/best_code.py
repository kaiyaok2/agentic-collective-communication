def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute coefficients and permutation mapping
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], dtype=x.dtype, device=x.device)
    perm = torch.tensor([(r + W // 2) % W for r in range(W)], dtype=torch.long, device=x.device)
    
    # Create index tensors for vectorized operations
    indices = torch.arange(W, device=x.device)
    
    # Round 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Rounds 2-8: Simplified operations
    for round_idx in range(7):
        # Vectorized scaling and permutation
        s_reshaped = s.view(W, S)
        buf_reshaped = torch.zeros_like(s_reshaped)
        
        for r in range(W):
            p = perm[r].item()
            buf_reshaped[p] = a[r] * s_reshaped[p] / W
        
        buf = buf_reshaped.view(-1)
        
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Inverse scaling (skip on last iteration)
        if round_idx < 6:
            s_reshaped = s.view(W, S)
            for r in range(W):
                p = perm[r].item()
                s_reshaped[p] = s_reshaped[p] / a[r]
            s = s_reshaped.view(-1)
    
    return s