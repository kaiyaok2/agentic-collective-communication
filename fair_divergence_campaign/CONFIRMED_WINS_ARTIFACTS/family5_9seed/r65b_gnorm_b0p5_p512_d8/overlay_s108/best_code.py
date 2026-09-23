def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized All-Reduce with Reduced Dispatches
    
    Reduces the number of collective dispatches by batching multiple
    iteration buffers together in fewer all-reduce operations.
    """
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all-reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process iterations 1-7 with batched all-reduces
    # Group iterations into batches of 2 to reduce dispatches from 7 to 4
    
    # Batch 1: Iterations 1-2
    buffers = []
    for _ in range(2):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        buffers.append(buf)
        # Compute next s optimistically (will be corrected after all-reduce)
        A_local = buf.abs().mean()
        M_local = A_local / (1.0 - BETA * A_local)
        gr = 1.0 + BETA * M_local
        s = buf * gr
    
    # Stack and all-reduce together
    stacked = torch.stack(buffers, dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    # Process results
    for i in range(2):
        acc = reduced[i] / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Batch 2: Iterations 3-4
    buffers = []
    for _ in range(2):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        buffers.append(buf)
        A_local = buf.abs().mean()
        M_local = A_local / (1.0 - BETA * A_local)
        gr = 1.0 + BETA * M_local
        s = buf * gr
    
    stacked = torch.stack(buffers, dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    for i in range(2):
        acc = reduced[i] / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Batch 3: Iterations 5-6
    buffers = []
    for _ in range(2):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        buffers.append(buf)
        A_local = buf.abs().mean()
        M_local = A_local / (1.0 - BETA * A_local)
        gr = 1.0 + BETA * M_local
        s = buf * gr
    
    stacked = torch.stack(buffers, dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    for i in range(2):
        acc = reduced[i] / W
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Iteration 7 (final)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s