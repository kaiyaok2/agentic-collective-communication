def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get the sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for vectorized operations across blocks
    s_view = s.view(B, S)
    
    # Iterations 1-6: Full iterations with scaling
    for iteration in range(6):
        # Vectorized computation of scaling factors for all blocks at once
        f = 1.0 + s_view.mean(dim=1).abs()  # Shape: (B,)
        
        # Apply scaling factors using broadcasting
        buf_view = s_view * f.unsqueeze(1)  # Broadcasting: (B,) -> (B, S)
        
        # All-reduce the scaled buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_view.view(-1))
        acc_view = acc.view(B, S)
        
        # Undo scaling and normalize using broadcasting
        s_view = acc_view / (world_size * f.unsqueeze(1))
    
    # Iteration 7: Compute factors, scale, and all-reduce
    f = 1.0 + s_view.mean(dim=1).abs()
    buf_view = s_view * f.unsqueeze(1)
    
    # Final all-reduce (iteration 7)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf_view.view(-1))
    
    # Final normalization (only by world_size, no factor undo)
    acc = acc / world_size
    
    return acc