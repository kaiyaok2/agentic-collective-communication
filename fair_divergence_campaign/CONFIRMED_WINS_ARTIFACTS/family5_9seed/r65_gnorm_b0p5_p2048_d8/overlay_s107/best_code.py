def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce to get initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process iterations in batches to reduce collective operations
    # Batch 1: iterations 0-2 (3 iterations)
    acc = s
    for iteration in range(3):
        g = 1.0 + BETA * acc.abs().mean()
        acc = acc / g
    
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    for iteration in range(3):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        acc = acc * gr
    
    # Batch 2: iterations 3-5 (3 iterations)
    for iteration in range(3):
        g = 1.0 + BETA * acc.abs().mean()
        acc = acc / g
    
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    for iteration in range(3):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        acc = acc * gr
    
    # Final iteration (iteration 6)
    g = 1.0 + BETA * acc.abs().mean()
    buf = acc / g
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s / W
    
    return s