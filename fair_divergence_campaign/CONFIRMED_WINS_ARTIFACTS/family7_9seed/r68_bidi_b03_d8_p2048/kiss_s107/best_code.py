
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to segments
    s_reshaped = s.view(W, S)
    
    # Create coefficient tensor
    coefs = torch.tensor([0.3 + 0.1*(r % 4) for r in range(W-1)] + [0.0], 
                         dtype=s.dtype, device=s.device).unsqueeze(1)
    
    # Shifted version
    s_next = torch.cat([s_reshaped[1:], s_reshaped[-1:]])
    
    # Combine
    buf = ((s_reshaped + coefs * s_next) / W).view(-1)
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
