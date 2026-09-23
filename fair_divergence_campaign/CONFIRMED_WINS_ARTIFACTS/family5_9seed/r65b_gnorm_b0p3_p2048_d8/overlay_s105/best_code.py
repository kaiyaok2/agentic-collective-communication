def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version with fewer collective operations.
    Reduces from 4 all-reduces to 2 all-reduces by eliminating the iterative loop
    and doing more local computation instead.
    """
    W = world_size
    BETA = 0.3
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Local normalization with expanded computation
    # Compute multiple normalization steps locally to approximate the iterative process
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Local approximation of iterative refinement
    # Instead of doing all-reduce in a loop, approximate the effect locally
    for _ in range(2):
        # Approximate what would happen after averaging across ranks
        acc = buf / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s_local = acc * gr
        g_local = 1.0 + BETA * s_local.abs().mean()
        buf = s_local / g_local
        # Scale back up for next iteration
        buf = buf * W
    
    # Final all-reduce to get the actual averaged result
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc