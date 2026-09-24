
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Pre-compute coefficients
    b_list = [0.15 + 0.1*(r % 3) for r in range(W)]
    b = torch.tensor(b_list, device=x.device, dtype=x.dtype)
    b_exp = b[:-1].unsqueeze(1)
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 iterations with backward sweep
    for _ in range(5):
        s_view = s.view(W, S)
        
        # Create buffer and apply forward sweep
        buf = s * inv_W
        buf_view = buf.view(W, S)
        buf_view[:-1] = (s_view[:-1] + b_exp * s_view[1:]) * inv_W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s_view = s.view(W, S)
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s_view[r] -= b_list[r] * s_view[r+1]
    
    # Final iteration
    s_view = s.view(W, S)
    buf = s * inv_W
    buf_view = buf.view(W, S)
    buf_view[:-1] = (s_view[:-1] + b_exp * s_view[1:]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
