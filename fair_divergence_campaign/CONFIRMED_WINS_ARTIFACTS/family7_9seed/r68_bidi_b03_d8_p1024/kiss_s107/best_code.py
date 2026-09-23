
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.3 + 0.1*(r % 5) for r in range(W)]
    
    # Initial synchronization
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to (W, S) for easier segment operations
    s_reshaped = s.view(W, S)
    
    # Do 6 complete iterations
    for _ in range(6):
        buf = s_reshaped.clone()
        
        # Forward sweep - vectorized where possible
        for r in range(W - 1):
            buf[r] = s_reshaped[r] + b[r] * s_reshaped[r + 1]
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            buf[r] = buf[r] - b[r] * buf[r + 1]
        
        s_reshaped = buf
    
    # Final forward sweep
    for r in range(W - 1):
        s_reshaped[r] = s_reshaped[r] + b[r] * s_reshaped[r + 1]
    
    return s_reshaped.view(-1)
