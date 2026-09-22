def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get the sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorize all iterations to reduce dispatch overhead
    # Reshape to work with batches more efficiently
    s_reshaped = s.view(B, S)
    
    for iteration in range(6):
        # Compute scale factors for all batches at once (no .item() calls)
        meansq = (s_reshaped ** 2).mean(dim=1)  # Shape: (B,)
        f = 1.0 + meansq  # Shape: (B,)
        
        # Apply scaling: multiply each batch by its factor
        scaled = s_reshaped * f.unsqueeze(1)  # Broadcasting: (B, 1) * (B, S)
        
        # Flatten back for all-reduce
        buf = scaled.view(-1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Reshape and unscale
        acc_reshaped = acc.view(B, S)
        s_reshaped = acc_reshaped / (world_size * f.unsqueeze(1))
    
    # Final iteration (iteration 6) - only divide by world_size
    meansq = (s_reshaped ** 2).mean(dim=1)
    f = 1.0 + meansq
    scaled = s_reshaped * f.unsqueeze(1)
    buf = scaled.view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc