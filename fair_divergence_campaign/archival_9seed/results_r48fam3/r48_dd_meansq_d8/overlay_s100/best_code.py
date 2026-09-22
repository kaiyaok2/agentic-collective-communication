def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process all 7 dependent iterations
    # Key insight: we can't reduce number of all-reduces due to dependencies,
    # but we can reduce local operations and cloning overhead
    
    for iteration in range(7):
        # Compute all factors at once for all blocks
        factors = torch.tensor([1.0 + (s[b*S:(b+1)*S]**2).mean().item() for b in range(B)], dtype=dtype)
        
        # Apply factors using vectorized operations where possible
        # Create a view/reshape to apply factors efficiently
        s_reshaped = s.view(B, S)
        factors_expanded = factors.view(B, 1)
        buf = (s_reshaped * factors_expanded).view(-1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by world_size and factors in one go
        if iteration < 6:  # For iterations 0-6, apply the inverse factor division
            acc_reshaped = acc.view(B, S)
            factors_expanded = factors.view(B, 1)
            s = (acc_reshaped / (world_size * factors_expanded)).view(-1)
        else:  # Final iteration
            s = acc / world_size
    
    return s