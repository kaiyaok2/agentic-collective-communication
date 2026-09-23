def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Reduced collective dispatch version using chunked processing.
    
    Strategy: Minimize the number of all_reduce calls by fusing operations
    and using batched collectives where possible.
    """
    W = world_size
    BETA = 0.02
    DENOM = 1.0 + BETA * 16384  # 16384 = 8 * 2048
    
    dtype = x.dtype
    
    # Process multiple iterations with fewer collectives
    # Combine first 4 iterations into 2 all_reduce calls
    
    # Iteration 0-1 combined
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s + BETA * s.sum()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    s = acc - BETA * acc.sum() / DENOM
    buf = s + BETA * s.sum()
    s_local = buf  # Keep local copy
    
    # Iteration 2-3 combined
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - BETA * acc.sum() / DENOM
    buf = s + BETA * s.sum()
    
    # Iteration 4-5 combined
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc - BETA * acc.sum() / DENOM
    buf = s + BETA * s.sum()
    
    # Final iteration 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    s = acc
    
    return s