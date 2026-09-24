
def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    N = 4096
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns and scaling
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    inv_W = 1.0 / W
    inv_N = 1.0 / N
    
    # 6 full iterations
    for _ in range(6):
        # Use sum instead of mean and pre-multiply inverse
        vs_sum = (v * s).sum() * inv_N
        buf = s + u * vs_sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        vacc_sum = (v * acc).sum() * inv_N
        s = acc - u * vacc_sum
    
    # Final iteration
    vs_sum = (v * s).sum() * inv_N
    buf = s + u * vs_sum
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
