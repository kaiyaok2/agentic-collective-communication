
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_reshaped = s.view(W, S)
    
    # Create coefficient tensor with shape (W-1, 1) for broadcasting
    b = torch.tensor([[0.15 + 0.1*(r % 3)] for r in range(W-1)], 
                     device=x.device, dtype=x.dtype)
    
    # Vectorized transformation
    result_reshaped = torch.zeros_like(s_reshaped)
    result_reshaped[:-1] = (s_reshaped[:-1] + b * s_reshaped[1:]) / W
    result_reshaped[-1] = s_reshaped[-1] / W
    
    return xm.all_reduce(xm.REDUCE_SUM, result_reshaped.view(-1))
