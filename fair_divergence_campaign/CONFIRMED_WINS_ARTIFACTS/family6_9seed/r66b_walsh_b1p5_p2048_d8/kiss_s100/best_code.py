
def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    beta_u = (1.5 - 3.0 * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / W
    
    # Iteration 1
    vs = v * s
    s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * vs.mean()) * inv_W - beta_u * (v * s).mean()
    
    # Iteration 2
    s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W - beta_u * (v * (xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W)).mean()
    
    # ... rest
    s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
    
    return s
