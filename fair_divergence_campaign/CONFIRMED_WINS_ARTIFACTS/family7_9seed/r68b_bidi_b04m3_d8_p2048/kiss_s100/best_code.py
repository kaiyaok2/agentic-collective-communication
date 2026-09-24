
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b_list = [0.4 + 0.08*(r % 3) for r in range(W)]
    b_tensor = torch.tensor(b_list[:-1], device=x.device, dtype=x.dtype).view(-1, 1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 full iterations with backward sweep
    for _ in range(5):
        s_view = s.view(W, S)
        buf = s / W
        buf_view = buf.view(W, S)
        
        # Forward sweep - vectorized
        buf_view[:-1] = (s_view[:-1] + b_tensor * s_view[1:]) / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s_view = s.view(W, S)
        
        # Backward sweep - vectorized
        for r in range(W - 2, -1, -1):
            s_view[r] = s_view[r] - b_list[r] * s_view[r+1]
    
    # Final forward+reduce without backward
    s_view = s.view(W, S)
    buf = s / W
    buf_view = buf.view(W, S)
    buf_view[:-1] = (s_view[:-1] + b_tensor * s_view[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
