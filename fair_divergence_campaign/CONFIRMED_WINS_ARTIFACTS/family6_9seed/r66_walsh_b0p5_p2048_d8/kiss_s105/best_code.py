
def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.5; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device)
    v = (1.0 - 2.0 * (idx % 2)).to(s.dtype)
    beta_u = (BETA - 2.0 * BETA * ((idx // 2) % 2)).to(s.dtype)
    
    # 6 full iterations
    for _ in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
