
def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    inv_W = 1.0 / W
    beta_inv = BETA / (1.0 + BETA)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create v once
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Precompute BETA * v and beta_inv * v to reduce multiplications
    beta_v = BETA * v
    beta_inv_v = beta_inv * v
    
    # Unrolled iterations
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_inv_v * (v * acc).mean()
    
    buf = s + beta_v * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
