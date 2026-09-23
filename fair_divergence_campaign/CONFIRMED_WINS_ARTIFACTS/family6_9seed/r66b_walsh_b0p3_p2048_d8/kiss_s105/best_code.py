
def r66b_walsh_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3; N = 16384
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_u = (BETA * (1 - 2 * ((idx // 2) % 2))).to(x.dtype)
    
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
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return acc
