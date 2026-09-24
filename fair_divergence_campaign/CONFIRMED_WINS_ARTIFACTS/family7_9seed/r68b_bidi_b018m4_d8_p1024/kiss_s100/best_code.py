def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape
    s_view = s.view(W, S)
    
    # Create weights and apply
    b = torch.tensor([0.18 + 0.13*(r % 4) for r in range(W-1)] + [0.0], 
                     device=s.device, dtype=s.dtype).view(W, 1)
    next_chunk = torch.cat([s_view[1:], s_view[-1:]], dim=0)
    
    buf = (s_view + b * next_chunk).view(-1) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
