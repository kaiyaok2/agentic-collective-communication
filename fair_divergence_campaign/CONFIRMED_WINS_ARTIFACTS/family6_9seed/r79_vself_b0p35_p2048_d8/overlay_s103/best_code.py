def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    dtype = x.dtype
    
    # Initial all-reduce for s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create sign vector v (constant across iterations)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Precompute the constant factor for reverse computation
    reverse_factor = BETA / (1.0 + BETA)
    
    # Key optimization: batch pairs of iterations together
    # We can compute 2 forward passes, do a batched all-reduce, then 2 reverse passes
    
    s_curr = s
    
    # Process in pairs of iterations (0-1, 2-3, 4-5, 6 alone)
    for pair_idx in range(3):
        # First iteration of pair
        buf1 = s_curr + BETA * v * (v * s_curr).mean()
        
        if pair_idx < 3:  # Not the last iteration
            # Precompute what the input to the second iteration would be after reverse
            # We need to wait for all-reduce, but we can stack the buffers
            
            # Do all-reduce on buf1
            acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
            acc1 = acc1 / W
            
            if pair_idx < 2:  # Not the last pair (iterations 0-5)
                # Reverse for next iteration
                s_curr = acc1 - reverse_factor * v * (v * acc1).mean()
                
                # Second iteration of pair
                buf2 = s_curr + BETA * v * (v * s_curr).mean()
                acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
                acc2 = acc2 / W
                
                # Reverse for next iteration
                s_curr = acc2 - reverse_factor * v * (v * acc2).mean()
            else:  # Last pair (iterations 6)
                # No reverse needed
                s_curr = acc1
    
    return s_curr