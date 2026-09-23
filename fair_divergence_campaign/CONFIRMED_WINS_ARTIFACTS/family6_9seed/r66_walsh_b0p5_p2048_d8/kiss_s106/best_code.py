
def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    N = 16384
    
    # Initial all-reduce - after this, all ranks have the same value
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh patterns
    idx = torch.arange(N, device=x.device, dtype=x.dtype)
    v = 1.0 - 2.0 * (idx % 2)
    u = 1.0 - 2.0 * ((idx // 2) % 2)
    
    # Precompute constant factors
    beta_u = BETA * u
    # Note: (v * u) has pattern [1,-1,-1,1,...] with mean 0
    # So v_beta_u_factor = 1.0 + BETA * 0 = 1.0
    
    # All computation is local since all ranks have identical values
    for _ in range(6):
        mean_s = (v * s).mean()
        buf = s + beta_u * mean_s
        # mean_buf = mean_s * 1.0 = mean_s (simplified!)
        s = buf - beta_u * mean_s
    
    # Final iteration (no subtraction step)
    s = s + beta_u * (v * s).mean()
    
    return s
