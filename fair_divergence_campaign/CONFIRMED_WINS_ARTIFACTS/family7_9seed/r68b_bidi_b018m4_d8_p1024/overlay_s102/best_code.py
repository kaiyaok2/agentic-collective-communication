def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute b coefficients
    b = [0.18 + 0.13 * (r % 4) for r in range(W)]
    
    # Stage 1: Initial all-reduce and coupling
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    for r in range(W - 1):
        start = r * S
        end = (r + 1) * S
        next_end = (r + 2) * S
        buf[start:end] = buf[start:end] + b[r] * buf[end:next_end]
    
    # Stages 2-7: Unroll and batch operations more efficiently
    for stage in range(2, 8):
        # All-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Uncouple, divide, and recouple in single pass
        buf = s / W
        
        # Uncouple previous coupling (backward pass)
        for r in range(W - 2, -1, -1):
            start = r * S
            end = (r + 1) * S
            next_end = (r + 2) * S
            buf[start:end] = buf[start:end] - b[r] * buf[end:next_end]
        
        # Apply new coupling (forward pass)
        for r in range(W - 1):
            start = r * S
            end = (r + 1) * S
            next_end = (r + 2) * S
            buf[start:end] = buf[start:end] + b[r] * buf[end:next_end]
    
    # Stage 8: Final all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s