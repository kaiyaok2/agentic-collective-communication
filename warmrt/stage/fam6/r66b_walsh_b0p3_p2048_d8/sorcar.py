
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3; N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute Walsh patterns
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    # Directly compute beta_u without intermediate u
    beta_u = (BETA - 2 * BETA * ((idx // 2) % 2)).to(x.dtype)
    
    # 6 complete iterations
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) / W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) / W
    
    return s
