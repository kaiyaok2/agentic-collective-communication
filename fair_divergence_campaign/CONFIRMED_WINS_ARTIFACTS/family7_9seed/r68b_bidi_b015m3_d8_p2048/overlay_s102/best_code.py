def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Pre-compute bidiagonal coefficients
    b = [0.15 + 0.1 * (r % 3) for r in range(W)]
    
    # Pre-allocate persistent buffers
    buf = torch.empty(W * S, dtype=dtype, device=x.device)
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Forward coupling, all-reduce, backward decoupling
    for stage in range(7):
        # Forward coupling: apply bidiagonal transform
        buf[:] = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r] * s[(r+1)*S:(r+2)*S]) / W
        
        # All-reduce dispatch
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward decoupling: undo the bidiagonal transform
        if stage < 6:  # Don't decouple after the last stage
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
    
    return s