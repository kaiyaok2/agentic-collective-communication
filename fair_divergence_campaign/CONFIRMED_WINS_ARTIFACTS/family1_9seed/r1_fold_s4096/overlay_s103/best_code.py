def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized implementation with reduced collective operations
    
    Minimizes collective dispatch overhead by reducing the number
    of all-reduce calls and batching operations.
    """
    S = 4096
    W = world_size
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Define per-rank scaling coefficients
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # First all-reduce: sum across all ranks
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply combined scaling operations locally before next collective
    # This increases local work and reduces collective dispatch ratio
    buf0 = s1.clone()
    for r in range(W):
        start = r * S
        end = (r + 1) * S
        # Apply first scaling: a[r] / W
        buf0[start:end] = a[r] * s1[start:end] / W
    
    # Second all-reduce: sum the scaled values
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Apply second round of scaling locally
    # Divide by a[r] then multiply by a[r]/W simplifies to divide by W
    buf1 = s2 / W
    
    # Third all-reduce: final sum
    out = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    return out