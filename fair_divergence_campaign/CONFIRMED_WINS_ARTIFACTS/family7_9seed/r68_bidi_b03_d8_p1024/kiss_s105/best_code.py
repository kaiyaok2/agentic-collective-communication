
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_view = s.view(W, S)
    
    # Use arange to generate coefficients
    r_idx = torch.arange(W-1, device=x.device, dtype=x.dtype)
    b = (0.3 + 0.1 * (r_idx % 5)).unsqueeze(1)
    
    buf = s / W
    buf.view(W, S)[:-1] = (s_view[:-1] + b * s_view[1:]) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
