def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    idx = torch.arange(16384, device=x.device)
    v = 1.0 - 2.0 * (idx % 2).to(x.dtype)
    beta_u = 0.5 - (idx // 2 % 2).to(x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for i in range(7):
        s = s + beta_u * (v * s).mean()
        s = xm.all_reduce(xm.REDUCE_SUM, s) / world_size
        if i < 6:
            s = s - beta_u * (v * s).mean()
    
    return s
