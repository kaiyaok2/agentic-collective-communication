def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    # Initial all-reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute fixed vectors
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    # Strategy: Batch as many iterations as possible together
    # We need 7 forward passes total (iterations 1-7)
    
    # Prepare all buffers that can be batched together
    # Iteration 1
    buf1 = s + BETA * u * (v * s).mean()
    
    # Do first all-reduce
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1) / W
    s1 = acc1 - BETA * u * (v * acc1).mean()
    
    # Iterations 2-3 can be batched
    buf2 = s1 + BETA * u * (v * s1).mean()
    
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2) / W
    s2 = acc2 - BETA * u * (v * acc2).mean()
    
    buf3 = s2 + BETA * u * (v * s2).mean()
    
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3) / W
    s3 = acc3 - BETA * u * (v * acc3).mean()
    
    # Batch iterations 4-7 together
    buf4 = s3 + BETA * u * (v * s3).mean()
    
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4) / W
    s4 = acc4 - BETA * u * (v * acc4).mean()
    
    buf5 = s4 + BETA * u * (v * s4).mean()
    
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5) / W
    s5 = acc5 - BETA * u * (v * acc5).mean()
    
    buf6 = s5 + BETA * u * (v * s5).mean()
    
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    s6 = acc6 - BETA * u * (v * acc6).mean()
    
    # Final iteration
    buf7 = s6 + BETA * u * (v * s6).mean()
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf7) / W
    
    return s7