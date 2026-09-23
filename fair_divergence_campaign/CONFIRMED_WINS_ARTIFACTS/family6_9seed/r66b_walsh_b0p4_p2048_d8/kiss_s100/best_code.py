
def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.4; N = 16384
    
    # Precompute Walsh patterns
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    beta_u = BETA * u  # Precompute BETA * u
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 2
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 3
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 4
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 5
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 6
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_u * (v * acc).mean()
    
    # Iteration 7 (final)
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc
