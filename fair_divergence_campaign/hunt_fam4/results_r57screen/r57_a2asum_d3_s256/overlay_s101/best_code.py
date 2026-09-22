def r57_a2asum_d3_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 3
    dtype = x.dtype
    
    # Compute scaling factors
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Pre-compute the scaling factor to reduce operations
    scale_factor = a / u
    
    cur = x.to(dtype)
    
    # Fuse the scaling with the initial value to reduce one operation
    cur = cur * a
    
    for _t in range(D):
        # All-reduce with SUM operation
        cur = xm.all_reduce(xm.REDUCE_SUM, cur)
        
        # Apply scaling for next iteration (or final result)
        # For iterations 0 to D-2, we scale by (a/u) for the next iteration
        # For iteration D-1, we just divide by u for the final result
        if _t < D - 1:
            cur = cur * scale_factor
        else:
            cur = cur / u
    
    return cur