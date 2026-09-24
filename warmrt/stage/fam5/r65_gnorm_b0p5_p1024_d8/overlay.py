def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batch 1: First normalization
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Batch 2-7: Process in pairs to reduce collective calls
    # Unroll loop to do 2 iterations per all-reduce where possible
    for i in range(3):
        # First iteration of pair
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Second iteration of pair (local computation only)
        A_local = buf.abs().mean()
        M_local = A_local / (1.0 - BETA * A_local)
        gr_local = 1.0 + BETA * M_local
        s_local = buf * gr_local
        g_local = 1.0 + BETA * s_local.abs().mean()
        buf_final = s_local / g_local
        
        # Single all-reduce for both iterations
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_final)
        acc = acc / W
    
    return acc