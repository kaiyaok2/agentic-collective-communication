def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce to get the sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 6 iterations of: compute scaling factors, scale, all-reduce, unscale
    for iteration in range(6):
        # Compute scaling factors for all blocks at once (vectorized)
        s_reshaped = s.view(B, S)
        f = 1.0 + s_reshaped.mean(dim=1).abs()  # Shape: (B,)
        
        # Scale all blocks at once (vectorized)
        buf = s_reshaped * f.unsqueeze(1)  # Broadcasting
        buf = buf.view(-1)  # Flatten back
        
        # All-reduce the scaled buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale all blocks at once (vectorized)
        acc_reshaped = acc.view(B, S)
        s = (acc_reshaped / (world_size * f.unsqueeze(1))).view(-1)
    
    # Final iteration: compute scaling factors, scale, all-reduce, then divide by world_size
    s_reshaped = s.view(B, S)
    f = 1.0 + s_reshaped.mean(dim=1).abs()  # Shape: (B,)
    
    # Scale all blocks at once
    buf = s_reshaped * f.unsqueeze(1)
    buf = buf.view(-1)
    
    # All-reduce the scaled buffer
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final division by world_size only (no unscaling by f)
    acc = acc / world_size
    
    return acc