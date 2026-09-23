def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.3 + 0.1*(r % 4) for r in range(W)]
    dtype = x.dtype
    
    # Stage 1: Initial all_reduce without chunking
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute indices and coefficients for efficiency
    indices = [(r*S, (r+1)*S, (r+1)*S, (r+2)*S) for r in range(W-1)]
    inv_W = 1.0 / W
    
    # Precompute b values as tensor for vectorized operations
    b_tensor = torch.tensor(b, dtype=dtype, device=x.device)
    
    # Stages 2-8: Apply bidiagonal coupling, decouple previous, and reduce
    for stage in range(7):
        # Decouple previous stage (if not first iteration)
        if stage > 0:
            for r in range(W - 2, -1, -1):
                r_start, r_end, rp1_start, rp1_end = indices[r]
                s[r_start:r_end] = s[r_start:r_end] - b[r]*s[rp1_start:rp1_end]
        
        # Apply bidiagonal coupling - fuse division with coupling
        # Add more local computation to balance collective overhead
        buf = s.clone()
        
        # Process in vectorized manner where possible
        for r in range(W - 1):
            r_start, r_end, rp1_start, rp1_end = indices[r]
            buf[r_start:r_end] = (s[r_start:r_end] + b[r]*s[rp1_start:rp1_end]) * inv_W
        
        buf[(W-1)*S:W*S] = s[(W-1)*S:W*S] * inv_W
        
        # Single all_reduce per stage instead of chunked
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s