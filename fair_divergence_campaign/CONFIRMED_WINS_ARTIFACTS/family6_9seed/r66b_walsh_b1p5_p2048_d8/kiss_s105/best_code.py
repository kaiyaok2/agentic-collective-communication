
def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    beta_u = BETA * u
    
    # 6 full forward-backward iterations
    for _ in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        s = acc - beta_u * (v * acc).mean()
    
    # Final forward step without backward
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
