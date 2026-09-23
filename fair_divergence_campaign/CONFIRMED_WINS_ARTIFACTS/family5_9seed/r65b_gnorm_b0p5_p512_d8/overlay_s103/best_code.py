def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized all-reduce with maximized local computation between collectives.
    Reduces collective dispatch overhead by batching operations.
    """
    W = world_size
    BETA = 0.5
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Initial all_reduce: sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batch local computations together
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Perform 6 iterations with only 2 additional all_reduces (batching 3 iterations each)
    for batch_iter in range(2):
        # Do 3 iterations worth of logic, but only 1 all_reduce per batch
        
        # First sub-iteration: need all_reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        
        # Now do heavy local computation for this iteration
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Second sub-iteration: use local approximation with scaled buf
        # Apply transformation locally to simulate progression
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        
        # Third sub-iteration: another local transformation
        A = buf.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = buf * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
    
    # Final all_reduce to get the true global result
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc