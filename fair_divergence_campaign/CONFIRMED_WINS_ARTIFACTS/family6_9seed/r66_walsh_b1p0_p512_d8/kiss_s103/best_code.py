
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 4096
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns (BETA=1.0, so u is the pattern)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / W
    
    # 6 full iterations
    for _ in range(6):
        buf = s + u * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - u * (v * acc).mean()
    
    # Final iteration (no inverse)
    buf = s + u * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
