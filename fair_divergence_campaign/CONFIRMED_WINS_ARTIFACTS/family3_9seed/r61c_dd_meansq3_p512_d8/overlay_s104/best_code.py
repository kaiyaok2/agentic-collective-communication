def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get the global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process 7 dependent iterations sequentially
    # Each iteration: compute factors, multiply, all-reduce, divide
    
    for iteration in range(7):
        # Compute all factors for this iteration
        factors = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            factor = 1.0 + 3.0*(sb*sb).mean()
            factors.append(factor)
        
        # Apply factors to all blocks
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * factors[b]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by world_size and factors (combined into final step)
        if iteration < 6:
            # For iterations 0-5, we need to divide by both world_size and the factors
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * factors[b])
        else:
            # For iteration 6 (last), just divide by world_size
            acc = acc / world_size
        
        s = acc
    
    return s