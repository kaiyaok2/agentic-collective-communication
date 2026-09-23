
def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute pattern vectors
    idx = torch.arange(N, device=x.device)
    v = (1.0 - 2.0 * (idx % 2)).to(x.dtype)
    beta_u = (1.5 - 3.0 * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / world_size
    
    # Perform 6 full iterations
    for _ in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration (no inverse transform)
    buf = s + beta_u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
