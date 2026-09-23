
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 16384
    inv_W = 1.0 / W
    beta_factor = 0.5
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute v once - alternating pattern
    v = (1.0 - 2.0 * (torch.arange(N, device=x.device) % 2)).to(x.dtype)
    
    # Iteration 1
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 2  
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 3
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 4
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 5
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 6
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc - beta_factor * v * (v * acc).mean()
    
    # Iteration 7 - final iteration, no correction needed
    vs_mean = (v * s).mean()
    buf = s + v * vs_mean
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
