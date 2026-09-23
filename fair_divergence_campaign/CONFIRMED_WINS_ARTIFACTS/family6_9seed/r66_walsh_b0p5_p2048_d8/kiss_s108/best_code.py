
def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    N = 16384
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute patterns
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Pre-compute BETA * u to avoid repeated multiplication
    beta_u = BETA * u
    inv_W = 1.0 / W
    
    # Perform 6 full iterations
    for i in range(6):
        buf = s + beta_u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration (no subtraction)
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return acc
