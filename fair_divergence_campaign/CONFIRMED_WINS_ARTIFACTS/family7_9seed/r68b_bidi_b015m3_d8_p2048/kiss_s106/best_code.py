
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_reshaped = s.view(W, S)
    
    # Build coefficient tensor with shape (W-1, 1)
    b_list = [[0.15 + 0.1*(r % 3)] for r in range(W-1)]
    b_tensor = torch.tensor(b_list, device=x.device, dtype=x.dtype)
    
    # Forward pass
    buf = s_reshaped.clone()
    buf[:-1] = s_reshaped[:-1] + b_tensor * s_reshaped[1:]
    buf = buf / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return s
