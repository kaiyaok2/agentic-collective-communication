
def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.7
    N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_u = (BETA - 2 * BETA * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / world_size
    
    # 6 full iterations
    for _ in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
