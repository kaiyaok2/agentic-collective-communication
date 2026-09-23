
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 1.0; N = 4096
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Iteration 1
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 2
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 3
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 4
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 5
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 6
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - u * (v * acc).mean()
    
    # Iteration 7
    buf = s + u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc
    
    return s
