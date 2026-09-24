
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor reshaped for 2D operations
    b_coef = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).view(-1, 1).expand(-1, S)
    inv_W = 1.0 / W
    
    s = s.view(W, S)
    
    # 6 full iterations
    for _ in range(6):
        # Forward sweep
        buf = s * inv_W
        buf[:-1] = (s[:-1] + b_coef * s[1:]) * inv_W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s[r] = s[r] - b[r] * s[r+1]
    
    # Final iteration without backward sweep
    buf = s * inv_W
    buf[:-1] = (s[:-1] + b_coef * s[1:]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return s
