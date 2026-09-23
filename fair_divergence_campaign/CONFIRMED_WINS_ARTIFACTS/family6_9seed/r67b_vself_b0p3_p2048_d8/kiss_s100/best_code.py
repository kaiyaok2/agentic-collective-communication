def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    beta_adj = BETA / (1.0 + BETA)
    inv_W = 1.0 / W
    
    # Create alternating pattern vector once
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 refinement iterations
    for i in range(6):
        v_dot_s = (v * s).mean()
        buf = s + BETA * v * v_dot_s
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        
        if i < 5:
            # Full iteration with correction step
            v_dot_acc = (v * acc).mean()
            s = acc - beta_adj * v * v_dot_acc
        else:
            # Final iteration - just use acc directly
            s = acc
    
    return s