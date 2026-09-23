
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    beta_u = BETA * u  # Precompute BETA * u
    
    # 6 iterations with reverse transform
    for _ in range(6):
        vs_mean = (v * s).mean()
        buf = s + beta_u * vs_mean
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration without reverse transform
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return s
