
def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.7
    N = 16384
    inv_W = 1.0 / world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns
    idx = torch.arange(N, device=x.device)
    v = (1.0 - 2.0 * (idx % 2)).to(s.dtype)
    beta_u = (BETA - 2.0 * BETA * ((idx // 2) % 2)).to(s.dtype)
    
    # 6 complete iterations
    for _ in range(6):
        mean_vs = (v * s).mean()
        s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * mean_vs) * inv_W
        s = s - beta_u * (v * s).mean()
    
    # Final iteration
    return xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
