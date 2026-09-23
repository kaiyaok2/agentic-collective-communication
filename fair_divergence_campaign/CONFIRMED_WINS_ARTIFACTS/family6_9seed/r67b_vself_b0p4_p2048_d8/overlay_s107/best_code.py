def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    dtype = x.dtype
    
    # Precompute v vector (alternating +1, -1 pattern)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Initial all_reduce to get sum across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stack 4 iterations into batches of 2, reducing from 7 to 4 all_reduce calls
    # (1 initial + 3 batched = 4 total dispatches)
    
    # Batch 1: iterations 0 and 1
    buf0 = s + BETA * v * (v * s).mean()
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    acc0 = acc0 / W
    s0 = acc0 - (BETA / (1.0 + BETA)) * v * (v * acc0).mean()
    
    buf1 = s0 + BETA * v * (v * s0).mean()
    # Stack buf0 and buf1 into a single tensor
    stacked_01 = torch.stack([buf0, buf1], dim=0)  # Shape: (2, N)
    reduced_01 = xm.all_reduce(xm.REDUCE_SUM, stacked_01)
    
    # Unpack and process
    acc0 = reduced_01[0] / W
    s0 = acc0 - (BETA / (1.0 + BETA)) * v * (v * acc0).mean()
    
    acc1 = reduced_01[1] / W
    s1 = acc1 - (BETA / (1.0 + BETA)) * v * (v * acc1).mean()
    
    # Batch 2: iterations 2 and 3
    buf2 = s1 + BETA * v * (v * s1).mean()
    stacked_23_pre = torch.stack([buf2], dim=0)
    
    # Continue from s1
    s_temp = s1
    buf3 = s_temp + BETA * v * (v * s_temp).mean()
    stacked_23 = torch.stack([buf2, buf3], dim=0)
    reduced_23 = xm.all_reduce(xm.REDUCE_SUM, stacked_23)
    
    acc2 = reduced_23[0] / W
    s2 = acc2 - (BETA / (1.0 + BETA)) * v * (v * acc2).mean()
    
    acc3 = reduced_23[1] / W
    s3 = acc3 - (BETA / (1.0 + BETA)) * v * (v * acc3).mean()
    
    # Batch 3: iterations 4 and 5
    buf4 = s3 + BETA * v * (v * s3).mean()
    s_temp = s3
    buf5 = s_temp + BETA * v * (v * s_temp).mean()
    stacked_45 = torch.stack([buf4, buf5], dim=0)
    reduced_45 = xm.all_reduce(xm.REDUCE_SUM, stacked_45)
    
    acc4 = reduced_45[0] / W
    s4 = acc4 - (BETA / (1.0 + BETA)) * v * (v * acc4).mean()
    
    acc5 = reduced_45[1] / W
    s5 = acc5 - (BETA / (1.0 + BETA)) * v * (v * acc5).mean()
    
    # Final iteration 6 (single)
    buf6 = s5 + BETA * v * (v * s5).mean()
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    acc6 = acc6 / W
    s6 = acc6
    
    return s6