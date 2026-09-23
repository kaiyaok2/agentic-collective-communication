
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    
    # Use arange for coefficient computation
    r_vals = torch.arange(W-1, device=x.device, dtype=x.dtype)
    b_vals = 0.3 + 0.1 * (r_vals % 4)
    
    left = s[:(W-1)*S].view(W-1, S)
    right = s[S:W*S].view(W-1, S)
    buf[:(W-1)*S] = (left + b_vals.unsqueeze(1) * right).view(-1) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
