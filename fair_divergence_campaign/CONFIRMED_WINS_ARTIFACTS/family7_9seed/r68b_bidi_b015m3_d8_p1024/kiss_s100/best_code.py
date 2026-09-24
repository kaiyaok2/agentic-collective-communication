def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b_list = [0.15 + 0.1*(r % 3) for r in range(W)]
    inv_W = 1.0 / W
    
    s_flat = xm.all_reduce(xm.REDUCE_SUM, x)
    s_2d = s_flat.view(W, S)
    
    # Precompute b tensor for vectorization
    b_tensor = torch.tensor(b_list[:-1], device=x.device, dtype=x.dtype).view(-1, 1)
    
    # 6 full iterations with backward sweep
    for iteration in range(6):
        buf = s_2d * inv_W
        # Vectorized forward sweep
        buf[0:W-1] = (s_2d[0:W-1] + b_tensor * s_2d[1:W]) * inv_W
        
        s_flat = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        s_2d = s_flat.view(W, S)
        
        # Vectorized backward sweep
        for r in range(W - 2, -1, -1):
            s_2d[r] = s_2d[r] - b_list[r] * s_2d[r + 1]
    
    # Final forward sweep and all-reduce (no backward)
    buf = s_2d * inv_W
    buf[0:W-1] = (s_2d[0:W-1] + b_tensor * s_2d[1:W]) * inv_W
    s_flat = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return s_flat