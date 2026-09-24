
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b_list = [0.22 + 0.11*(r % 5) for r in range(W)]
    b_tensor = torch.tensor(b_list[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)
    inv_W = 1.0 / W
    
    # Reshape input to 2D and work in 2D throughout
    s = xm.all_reduce(xm.REDUCE_SUM, x.view(W, S))
    buf = torch.zeros_like(s)
    
    # 5 complete iterations
    for _ in range(5):
        # Forward sweep
        buf[:-1] = (s[:-1] + b_tensor * s[1:]) * inv_W
        buf[-1] = s[-1] * inv_W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s[r] -= b_list[r] * s[r + 1]
    
    # Final incomplete iteration
    buf[:-1] = (s[:-1] + b_tensor * s[1:]) * inv_W
    buf[-1] = s[-1] * inv_W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf).view(-1)
