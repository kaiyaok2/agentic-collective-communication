
def r67_vself_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    BETA_FACTOR = BETA / (1.0 + BETA)
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Precompute constant vectors
    beta_v = BETA * v
    beta_factor_v = BETA_FACTOR * v
    
    # Iteration 1
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 2
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 3
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 4
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 5
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 6
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - beta_factor_v * (v * acc).mean()
    
    # Iteration 7 (final)
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s
