
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute coefficients
    b_list = [0.2 + 0.12 * (r % 4) for r in range(W - 1)]
    b_tensor = torch.tensor(b_list, device=x.device, dtype=x.dtype).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1 - forward sweep only
    buf = s / W
    s_view = s.view(W, S)
    buf_view = buf.view(W, S)
    buf_view[:-1] = (s_view[:-1] + b_tensor * s_view[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Iterations 2-7 - backward then forward sweep
    for _ in range(6):
        s_view = s.view(W, S)
        # Backward sweep - must be sequential
        for r in range(W - 2, -1, -1):
            s_view[r] -= b_list[r] * s_view[r + 1]
        
        # Forward sweep - vectorized
        buf = s / W
        buf_view = buf.view(W, S)
        buf_view[:-1] = (s_view[:-1] + b_tensor * s_view[1:]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
