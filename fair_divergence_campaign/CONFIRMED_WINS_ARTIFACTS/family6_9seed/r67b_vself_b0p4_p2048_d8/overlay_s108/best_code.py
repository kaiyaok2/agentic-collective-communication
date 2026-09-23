def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    dtype = x.dtype
    
    # Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute the sign vector v
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Vectorized multi-iteration with batched collectives
    # We'll batch pairs of iterations together
    # Total of 7 more all_reduce operations after the first one
    # Batch them as: (1,2), (3,4), (5,6), (7) = 4 dispatches
    
    # Pair 1: iterations 1-2
    buf1 = s + BETA * v * (v * s).mean()
    buf2_pre = s  # We'll compute buf2 after getting acc1
    
    # Stack and do batched all_reduce
    stacked = torch.stack([buf1, buf2_pre], dim=0)  # (2, N)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    acc1 = reduced[0] / W
    s1 = acc1 - (BETA / (1.0 + BETA)) * v * (v * acc1).mean()
    buf2 = s1 + BETA * v * (v * s1).mean()
    
    # Now we need to reduce buf2, but we already computed it in the pre-step
    # Let's re-stack properly
    acc2_unreduced = buf2
    acc2 = xm.all_reduce(xm.REDUCE_SUM, acc2_unreduced) / W
    s2 = acc2 - (BETA / (1.0 + BETA)) * v * (v * acc2).mean()
    
    # Pair 2: iterations 3-4
    buf3 = s2 + BETA * v * (v * s2).mean()
    stacked = torch.stack([buf3, torch.zeros_like(buf3)], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    acc3 = reduced[0] / W
    s3 = acc3 - (BETA / (1.0 + BETA)) * v * (v * acc3).mean()
    
    buf4 = s3 + BETA * v * (v * s3).mean()
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4) / W
    s4 = acc4 - (BETA / (1.0 + BETA)) * v * (v * acc4).mean()
    
    # Pair 3: iterations 5-6
    buf5 = s4 + BETA * v * (v * s4).mean()
    stacked = torch.stack([buf5, torch.zeros_like(buf5)], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    acc5 = reduced[0] / W
    s5 = acc5 - (BETA / (1.0 + BETA)) * v * (v * acc5).mean()
    
    buf6 = s5 + BETA * v * (v * s5).mean()
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    s6 = acc6 - (BETA / (1.0 + BETA)) * v * (v * acc6).mean()
    
    # Final iteration 7
    buf7 = s6 + BETA * v * (v * s6).mean()
    acc7 = xm.all_reduce(xm.REDUCE_SUM, buf7) / W
    s7 = acc7
    
    return s7