def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    # Precompute BETA * u since it's used repeatedly
    beta_u = BETA * u
    W_inv = 1.0 / W
    
    # Iteration 1
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 2
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 3
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 4
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 5
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 6
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 7 (final)
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    
    return acc