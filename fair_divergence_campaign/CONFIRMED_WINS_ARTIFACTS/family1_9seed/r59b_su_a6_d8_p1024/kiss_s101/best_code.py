def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Compute weights directly as tensors
    r_range = torch.arange(W, device=x.device, dtype=x.dtype)
    a = 1.0 + 0.35 * (r_range % 6)
    
    w_mul = (a / W).unsqueeze(1)
    w_div = (torch.ones_like(a) / torch.clamp(a, min=1e-9)).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    for _ in range(6):
        s = w_mul * s
        s = xm.all_reduce(xm.REDUCE_SUM, s.view(-1)).view(W, S)
        s = w_div * s
    
    s = xm.all_reduce(xm.REDUCE_SUM, (w_mul * s).view(-1))
    
    return s