def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(x.dtype).to(x.device)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype).to(x.device)
    beta_u = BETA * u
    
    # First 6 full iterations
    for _ in range(6):
        vs_mean = (v * s).mean()
        buf = s + beta_u * vs_mean
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        vacc_mean = (v * acc).mean()
        s = acc - beta_u * vacc_mean
    
    # Last iteration
    vs_mean = (v * s).mean()
    buf = s + beta_u * vs_mean
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s