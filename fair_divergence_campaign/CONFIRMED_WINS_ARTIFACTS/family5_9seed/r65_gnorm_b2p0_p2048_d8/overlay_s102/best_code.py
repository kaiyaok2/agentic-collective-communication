def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # Allocate buffer to hold all 8 iteration states
    # Each iteration produces a vector of size 8*2048
    buffer = torch.zeros((8, x.shape[0]), dtype=dtype, device=x.device)
    
    # Iteration 0: Initial all-reduce of x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    buffer[0] = buf
    
    # Iterations 1-6: Intermediate iterations
    for i in range(1, 7):
        # Local computation using previous buffer state
        prev = buffer[i-1]
        A = prev.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s_local = prev * gr
        g_local = 1.0 + BETA * s_local.abs().mean()
        buf_local = s_local / g_local
        buffer[i] = buf_local
    
    # Iteration 7: Final iteration (no normalization needed, just store)
    prev = buffer[6]
    A = prev.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s_local = prev * gr
    g_local = 1.0 + BETA * s_local.abs().mean()
    buf_local = s_local / g_local
    buffer[7] = buf_local
    
    # Single concatenated all-reduce on entire buffer
    # Flatten buffer to 1D for all-reduce
    flat_buffer = buffer.reshape(-1)
    reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat_buffer)
    reduced_buffer = reduced_flat.reshape(8, x.shape[0])
    
    # Average by world_size and extract final result
    reduced_buffer = reduced_buffer / W
    
    # The final result is at index 7
    s = reduced_buffer[7]
    
    return s