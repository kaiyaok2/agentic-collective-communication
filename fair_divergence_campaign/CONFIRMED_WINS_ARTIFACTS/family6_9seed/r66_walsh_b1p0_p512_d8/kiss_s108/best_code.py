
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; N = 4096
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    W_inv = 1.0 / W
    
    # Unroll to minimize overhead
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    vs_mean = (v * s).mean()
    buf = s + u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    s = acc - u * (v * acc).mean()
    
    # Final iteration (no inverse)
    s = xm.all_reduce(xm.REDUCE_SUM, s + u * (v * s).mean()) * W_inv
    
    return s
