def r65_gnorm_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    dtype = x.dtype
    
    # First all-reduce: get initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute first g locally
    g = 1.0 + BETA * s.abs().mean()
    
    # Do all 6 iterations with all-reduces
    acc = s
    for iteration in range(6):
        # Fuse the division and all-reduce preparation
        buf = acc / g
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        
        # Fuse all local operations for next iteration
        if iteration < 5:
            # Compute all scalars in one go to maximize local work
            acc_abs = acc.abs()
            A = acc_abs.mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            acc = acc * gr
            
            # Pre-compute g for next iteration to fuse operations
            g = 1.0 + BETA * acc_abs.mean() * gr
        else:
            # Last iteration - no need to prepare for next
            pass
    
    return acc