
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # All_reduce and reshape in one go
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    # Build coefficient tensor with arange
    r_indices = torch.arange(W-1, device=x.device)
    b_coef = (0.3 + 0.1 * (r_indices % 5)).to(x.dtype).view(-1, 1)
    
    # Clone and apply
    buf = s.clone()
    buf[:-1] = (s[:-1] + b_coef * s[1:]) / W
    buf[-1] = s[-1] / W
    
    # Final all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
