
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    # Pre-compute scaling factors
    a = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], device=x.device, dtype=x.dtype)
    scale_fwd = (a / W).view(W, 1)
    a_clamped = a.clamp(min=1e-9)
    scale_inv = (torch.ones_like(a_clamped) / a_clamped).view(W, 1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for i in range(7):
        buf = (s.view(W, S) * scale_fwd).view(-1)
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        if i < 6:
            s = (s.view(W, S) * scale_inv).view(-1)
    
    return s
