
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    s_view = s.view(W, S)
    
    # Create b coefficients using arange for efficiency
    r_vals = torch.arange(W-1, device=x.device, dtype=x.dtype)
    b_vals = (0.35 + 0.09 * (r_vals % 4)).view(W-1, 1)
    
    # Vectorized computation
    buf_first = (s_view[:-1] + b_vals * s_view[1:]) / W
    buf_last = s_view[-1:] / W
    
    buf = torch.cat([buf_first, buf_last], dim=0).view(-1)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
