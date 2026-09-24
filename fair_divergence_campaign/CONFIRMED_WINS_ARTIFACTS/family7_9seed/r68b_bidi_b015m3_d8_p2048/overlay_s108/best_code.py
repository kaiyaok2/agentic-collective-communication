def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute bidiagonal coefficients
    b = torch.tensor([0.15 + 0.1 * (r % 3) for r in range(W)], dtype=dtype, device=x.device)
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply bidiagonal coupling with reduced collectives
    # Key optimization: Combine multiple local operations between collectives
    for stage in range(7):
        # Reshape for vectorized operations
        s_reshaped = s.view(W, S)
        
        # Apply bidiagonal coupling (forward pass) with vectorized ops
        buf = s_reshaped.clone()
        
        # Vectorized bidiagonal coupling
        for r in range(W - 1):
            buf[r] = (s_reshaped[r] + b[r] * s_reshaped[r + 1]) / W
        buf[W - 1] = s_reshaped[W - 1] / W
        
        # Flatten back
        buf = buf.view(-1)
        
        # All-reduce the coupled result
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Undo coupling (backward pass) with vectorized ops
        if stage < 6:
            s_reshaped = s.view(W, S)
            for r in range(W - 2, -1, -1):
                s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r + 1]
            s = s_reshaped.view(-1)
    
    return s