def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations of the dependent operation
    for iteration in range(6):
        # Fused computation: compute all 8 scaling factors in vectorized manner
        # Reshape to (B, S) for vectorized operations
        s_reshaped = s.view(B, S)
        
        # Vectorized abs and mean across all blocks at once
        abs_vals = s_reshaped.abs()
        means = abs_vals.mean(dim=1)  # Shape: (B,)
        f = 1.0 + means  # Shape: (B,)
        
        # Vectorized scaling: broadcast f to (B, S) and multiply
        f_expanded = f.view(B, 1)  # Shape: (B, 1)
        buf_reshaped = s_reshaped * f_expanded  # Shape: (B, S)
        buf = buf_reshaped.view(B * S)  # Flatten back
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Vectorized division: reshape, divide, and flatten
        acc_reshaped = acc.view(B, S)
        f_expanded = f.view(B, 1)
        s_reshaped = acc_reshaped / (world_size * f_expanded)
        s = s_reshaped.view(B * S)
    
    # Final iteration (7th) - same pattern but with final division by world_size
    s_reshaped = s.view(B, S)
    abs_vals = s_reshaped.abs()
    means = abs_vals.mean(dim=1)
    f = 1.0 + means
    f_expanded = f.view(B, 1)
    buf_reshaped = s_reshaped * f_expanded
    buf = buf_reshaped.view(B * S)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc