
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.2 + 0.1*(r % 3) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-allocate and fill
    buf = s / W  # Start with all elements divided
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    
    # Only update first W-1 segments
    s_next = s_reshaped[1:]
    s_curr = s_reshaped[:-1]
    b_tensor = torch.tensor(b[:(W-1)], device=s.device, dtype=s.dtype).view(-1, 1)
    
    buf_reshaped[:-1] = (s_curr + b_tensor * s_next) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
