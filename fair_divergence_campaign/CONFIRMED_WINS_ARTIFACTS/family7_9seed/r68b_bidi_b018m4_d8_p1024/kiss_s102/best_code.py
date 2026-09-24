
def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.18 + 0.13*(r % 4) for r in range(W)]
    
    # Create b tensor once for forward pass
    b_fwd = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).view(W-1, 1)
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations
    for _ in range(6):
        # Forward pass
        s_2d = s.view(W, S)
        buf = s * inv_W
        buf_2d = buf.view(W, S)
        buf_2d[:-1] = (s_2d[:-1] + b_fwd * s_2d[1:]) * inv_W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass
        s_2d = s.view(W, S)
        for r in range(W - 2, -1, -1):
            s_2d[r] -= b[r] * s_2d[r+1]
        s = s_2d.view(-1)
    
    # Final iteration
    s_2d = s.view(W, S)
    buf = s * inv_W
    buf_2d = buf.view(W, S)
    buf_2d[:-1] = (s_2d[:-1] + b_fwd * s_2d[1:]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
