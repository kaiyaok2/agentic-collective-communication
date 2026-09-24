
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    shifted = torch.zeros_like(s)
    shifted[:-1] = s[1:]
    
    # Use arange to create coefficients
    b = (0.15 + 0.1 * (torch.arange(W, device=x.device, dtype=x.dtype) % 3)).view(W, 1)
    
    return xm.all_reduce(xm.REDUCE_SUM, ((s + b * shifted) / W).view(-1))
