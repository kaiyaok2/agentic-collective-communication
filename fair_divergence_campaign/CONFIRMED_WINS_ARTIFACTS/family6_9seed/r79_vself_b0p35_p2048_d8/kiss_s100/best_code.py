
def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    BETA_RATIO = BETA / (1.0 + BETA)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create alternating vector: [1, -1, 1, -1, ...]
    N = x.numel()
    indices = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (indices % 2)).to(dtype=x.dtype)
    
    # 6 full iterations
    for _ in range(6):
        vs_mean = (v * s).mean()
        buf = s + (BETA * vs_mean) * v
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        va_mean = (v * acc).mean()
        s = acc - (BETA_RATIO * va_mean) * v
    
    # Final iteration without bias removal
    vs_mean = (v * s).mean()
    buf = s + (BETA * vs_mean) * v
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return acc
