
def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 16384
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device)
    v = (1.0 - 2.0 * (idx % 2)).to(x.dtype)
    beta_u = (0.5 - (idx // 2) % 2).to(x.dtype)
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_u * (v * acc).mean()
    
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
