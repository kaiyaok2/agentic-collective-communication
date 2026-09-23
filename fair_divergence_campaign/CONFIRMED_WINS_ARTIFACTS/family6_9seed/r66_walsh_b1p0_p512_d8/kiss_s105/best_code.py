def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 4096
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute patterns
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    # Iteration 1 (BETA * u simplified to u since BETA = 1.0)
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 2
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 3
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 4
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 5
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 6 - different pattern!
    buf = s + u * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 7 (partial)
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s
