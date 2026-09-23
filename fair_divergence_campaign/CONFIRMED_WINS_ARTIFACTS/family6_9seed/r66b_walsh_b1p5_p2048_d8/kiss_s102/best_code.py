
def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W_inv = 1.0 / world_size
    BETA = 1.5
    N = 16384
    
    # Create Walsh patterns
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    BETA_u = BETA * u  # Precompute this product
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6
    for _ in range(6):
        buf = s + BETA_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
        s = acc - BETA_u * (v * acc).mean()
    
    # Final iteration (no inverse transform)
    buf = s + BETA_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    
    return s
