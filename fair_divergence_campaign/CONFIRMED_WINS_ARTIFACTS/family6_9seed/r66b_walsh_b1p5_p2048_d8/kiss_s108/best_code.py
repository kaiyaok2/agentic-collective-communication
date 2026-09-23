def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = 16384
    inv_W = 1.0 / world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_u = (1.5 - 3.0 * ((idx // 2) % 2)).to(x.dtype)
    
    # Iteration 1
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 2
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 3
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 4
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 5
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 6
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    m = (v * acc).mean()
    s = acc - beta_u * m
    
    # Iteration 7
    m = (v * s).mean()
    buf = s + beta_u * m
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return acc