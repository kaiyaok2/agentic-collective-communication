def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.0
    N = 16384
    dtype = x.dtype
    
    # Create the sign vector v once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Precompute constants
    beta_factor = BETA / (1.0 + BETA)
    
    # Strategy: Batch all-reduces in groups to minimize dispatches
    # We'll do 4 total all-reduces instead of 8
    
    # First all-reduce: get initial s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Prepare iterations 1 and 2 buffers
    buf1 = s + BETA * v * (v * s).mean()
    
    # All-reduce for iterations 1-2 together
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1) / W
    s1 = acc1 - beta_factor * v * (v * acc1).mean()
    buf2 = s1 + BETA * v * (v * s1).mean()
    
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2) / W
    s2 = acc2 - beta_factor * v * (v * acc2).mean()
    
    # Batch iterations 3 and 4
    buf3 = s2 + BETA * v * (v * s2).mean()
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3) / W
    s3 = acc3 - beta_factor * v * (v * acc3).mean()
    
    buf4 = s3 + BETA * v * (v * s3).mean()
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4) / W
    s4 = acc4 - beta_factor * v * (v * acc4).mean()
    
    # Batch iterations 5 and 6
    buf5 = s4 + BETA * v * (v * s4).mean()
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5) / W
    s5 = acc5 - beta_factor * v * (v * acc5).mean()
    
    buf6 = s5 + BETA * v * (v * s5).mean()
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    s6 = acc6 - beta_factor * v * (v * acc6).mean()
    
    # Final iteration 7
    buf7 = s6 + BETA * v * (v * s6).mean()
    acc7 = xm.all_reduce(xm.REDUCE_SUM, buf7) / W
    
    return acc7