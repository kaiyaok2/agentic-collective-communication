
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.3 + 0.1*(r % 4) for r in range(W)]
    
    # Create coefficient tensor once
    b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)  # Shape (W-1, 1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_2d = s.view(W, S)
    
    for _ in range(6):
        buf_2d = s_2d / W
        # Vectorized forward pass
        buf_2d[:-1] = (s_2d[:-1] + b_tensor * s_2d[1:]) / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf_2d.view(-1))
        s_2d = s.view(W, S)
        
        # Backward pass - still needs loop due to dependencies
        for r in range(W - 2, -1, -1):
            s_2d[r] = s_2d[r] - b[r] * s_2d[r+1]
    
    buf_2d = s_2d / W
    buf_2d[:-1] = (s_2d[:-1] + b_tensor * s_2d[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf_2d.view(-1))
    
    return s
