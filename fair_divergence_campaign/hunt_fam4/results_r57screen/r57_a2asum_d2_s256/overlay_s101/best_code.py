def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 2
    
    # Compute scale factor a for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute normalization factor u
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Fused scale factor: a/u
    fused_scale = a / u
    
    # Preserve input dtype
    dtype = x.dtype
    
    cur = x
    
    for stage in range(D):
        # Apply fused scale factor (a/u) at the beginning of each stage
        scaled_cur = fused_scale * cur
        
        # Replace the all-to-all + sum + all-gather pattern with a single all_reduce
        # The all-to-all pattern was:
        # 1. Reshape into blocks
        # 2. Gather all blocks
        # 3. Extract relevant blocks and sum
        # 4. All-gather the result
        # 
        # This is equivalent to: all_reduce with SUM operation
        # Each rank contributes its scaled_cur, and all ranks get the sum
        cur = xm.all_reduce(xm.REDUCE_SUM, scaled_cur)
    
    return cur