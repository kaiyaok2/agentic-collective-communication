
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.22 + 0.11*(r % 5) for r in range(W)]
    
    # Work in 2D from the start
    x_2d = x.reshape(W, S)
    s_2d = xm.all_reduce(xm.REDUCE_SUM, x_2d)
    
    # Pre-compute b tensor for vectorized operations
    b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).reshape(W-1, 1)
    inv_W = 1.0 / W
    
    # First forward sweep (no backward before it)
    buf_2d = s_2d * inv_W
    buf_2d[:-1] = (s_2d[:-1] + b_tensor * s_2d[1:]) * inv_W
    
    s_2d = xm.all_reduce(xm.REDUCE_SUM, buf_2d)
    
    # Next 6 iterations: backward sweep then forward sweep
    for iteration in range(6):
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s_2d[r] -= b[r] * s_2d[r+1]
        
        # Forward sweep
        buf_2d = s_2d * inv_W
        buf_2d[:-1] = (s_2d[:-1] + b_tensor * s_2d[1:]) * inv_W
        
        s_2d = xm.all_reduce(xm.REDUCE_SUM, buf_2d)
    
    return s_2d.reshape(-1)
