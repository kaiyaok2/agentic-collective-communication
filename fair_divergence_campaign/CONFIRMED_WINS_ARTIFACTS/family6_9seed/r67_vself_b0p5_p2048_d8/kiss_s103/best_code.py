
def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.5; N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating vector
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    # Precompute scaled versions
    v_beta = BETA * v
    v_beta_adj = (BETA / 1.5) * v
    
    # Iterations 1-6
    for _ in range(6):
        vs_mean = (v * s).mean()
        buf = s + v_beta * vs_mean
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        va_mean = (v * acc).mean()
        s = acc - v_beta_adj * va_mean
    
    # Final iteration
    vs_mean = (v * s).mean()
    buf = s + v_beta * vs_mean
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
