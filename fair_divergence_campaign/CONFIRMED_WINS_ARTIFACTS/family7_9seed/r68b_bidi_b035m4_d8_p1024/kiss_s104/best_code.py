
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Create coefficient vector more efficiently
    b_fwd = torch.tensor([0.35 + 0.09*(r % 4) for r in range(W-1)], 
                         device=x.device, dtype=x.dtype).repeat_interleave(S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 full iterations
    for _ in range(5):
        # Vectorized forward sweep
        buf = s / W
        buf[:(W-1)*S] = (s[:(W-1)*S] + b_fwd * s[S:]) / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            b_r = 0.35 + 0.09*(r % 4)
            offs = r * S
            s[offs:offs+S] = s[offs:offs+S] - b_r * s[offs+S:offs+2*S]
    
    # Final forward iteration
    buf = s / W
    buf[:(W-1)*S] = (s[:(W-1)*S] + b_fwd * s[S:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
