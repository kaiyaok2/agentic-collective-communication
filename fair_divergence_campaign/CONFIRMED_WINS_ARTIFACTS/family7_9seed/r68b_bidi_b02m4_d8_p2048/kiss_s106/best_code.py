
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    inv_W = 1.0 / W
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    segments = s.view(W, S)
    
    # Create b pattern that cycles every 4
    b_vals = ((0.2 + 0.12 * (torch.arange(W-1, device=x.device, dtype=x.dtype) % 4)) * inv_W).view(-1, 1)
    
    result = torch.cat([
        (segments[:-1] * inv_W + segments[1:] * b_vals).reshape(-1),
        (segments[-1] * inv_W)
    ])
    
    return xm.all_reduce(xm.REDUCE_SUM, result)
