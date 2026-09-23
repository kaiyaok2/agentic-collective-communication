
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute coefficients
    b_list = [0.2 + 0.1*(r % 3) for r in range(W)]
    b_fwd = torch.tensor(b_list[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_reshaped = s.view(W, S)
    
    # Iteration 1
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b_list[r] * s_reshaped[r + 1]
    
    # Iteration 2
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b_list[r] * s_reshaped[r + 1]
    
    # Iteration 3
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b_list[r] * s_reshaped[r + 1]
    
    # Iteration 4
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b_list[r] * s_reshaped[r + 1]
    
    # Iteration 5
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b_list[r] * s_reshaped[r + 1]
    
    # Iteration 6
    buf = s_reshaped / W
    buf[:-1] = (s_reshaped[:-1] + b_fwd * s_reshaped[1:]) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return s
