
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute Walsh patterns once with proper device/dtype
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1.0 - 2.0 * (idx % 2)).to(x.dtype)
    u = (1.0 - 2.0 * ((idx // 2) % 2)).to(x.dtype)
    beta_u = BETA * u
    
    # 6 full iterations
    for _ in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        s = acc - beta_u * (v * acc).mean()
    
    # Last iteration
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
