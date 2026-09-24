
def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.35 + 0.1*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute outside loop
    b_fwd_2d = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)
    inv_W = 1.0 / W
    
    for iter_idx in range(7):
        # Forward pass
        s_view = s.view(W, S)
        buf_view = torch.zeros_like(s_view)
        buf_view[:-1] = (s_view[:-1] + b_fwd_2d * s_view[1:]) * inv_W
        buf_view[-1] = s_view[-1] * inv_W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf_view.reshape(-1))
        
        # Backward pass (skip on last iteration)
        if iter_idx < 6:
            s_view = s.view(W, S)
            for r in range(W - 2, -1, -1):
                s_view[r] = s_view[r] - b[r] * s_view[r + 1]
            s = s_view.view(-1)
    
    return s
