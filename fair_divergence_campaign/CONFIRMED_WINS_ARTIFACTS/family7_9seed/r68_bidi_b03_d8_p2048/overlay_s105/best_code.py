def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Precompute b coefficients as a tensor for vectorization
    b_vals = torch.tensor([0.3 + 0.1 * (r % 4) for r in range(W-1)], 
                          dtype=dtype, device=x.device)
    
    # Step 1: Single all_reduce to gather full initial vector
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for easier vectorized operations
    result = s.view(W, S)
    
    # Process 7 intermediate stages
    for stage in range(7):
        # Apply bidiagonal coupling - vectorized
        # buf[r] = (result[r] + b[r]*result[r+1]) / W for r < W-1
        # buf[W-1] = result[W-1] / W
        
        buf = result.clone()
        buf[:-1] = (result[:-1] + b_vals.unsqueeze(1) * result[1:]) / W
        buf[-1] = result[-1] / W
        
        # After all_reduce, we'd sum W copies, so multiply back
        result = buf * W
        
        # Undo coupling - vectorized backward pass
        # result[r] = result[r] - b[r]*result[r+1] for r in reverse
        for r in range(W - 2, -1, -1):
            result[r] = result[r] - b_vals[r] * result[r+1]
    
    # Final coupling without undo
    buf = result.clone()
    buf[:-1] = (result[:-1] + b_vals.unsqueeze(1) * result[1:]) / W
    buf[-1] = result[-1] / W
    result = buf * W
    
    # Flatten back
    return result.view(W * S)