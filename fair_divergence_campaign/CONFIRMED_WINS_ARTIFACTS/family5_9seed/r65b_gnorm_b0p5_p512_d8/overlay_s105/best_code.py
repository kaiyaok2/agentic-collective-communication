def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Pipelined dual-buffer all-reduce chain strategy.
    
    Overlaps computation of normalization scalars for iteration i with 
    all-reduce communication for iteration i+1 by maintaining two buffers 
    and alternating between them.
    """
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initialize two buffers for pipelining
    buffer_a = x.clone()
    buffer_b = torch.zeros_like(x)
    
    # First iteration (no overlap possible)
    s = xm.all_reduce(xm.REDUCE_SUM, buffer_a)
    g = 1.0 + BETA * s.abs().mean()
    buffer_a = s / g
    
    # Start all-reduce for iteration 1 in buffer_a
    acc_a = xm.all_reduce(xm.REDUCE_SUM, buffer_a)
    acc_a = acc_a / W
    
    # Iterations 1-6: pipeline compute and communication
    for i in range(6):
        if i % 2 == 0:
            # Compute on buffer_a while buffer_b communicates
            A = acc_a.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s = acc_a * gr
            g = 1.0 + BETA * s.abs().mean()
            buffer_b = s / g
            
            # Start all-reduce on buffer_b
            acc_b = xm.all_reduce(xm.REDUCE_SUM, buffer_b)
            acc_b = acc_b / W
        else:
            # Compute on buffer_b while buffer_a communicates
            A = acc_b.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s = acc_b * gr
            g = 1.0 + BETA * s.abs().mean()
            buffer_a = s / g
            
            # Start all-reduce on buffer_a
            acc_a = xm.all_reduce(xm.REDUCE_SUM, buffer_a)
            acc_a = acc_a / W
    
    # Final iteration (iteration 6): no all-reduce needed after this
    A = acc_b.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc_b * gr
    g = 1.0 + BETA * s.abs().mean()
    buffer_a = s / g
    
    # Final all-reduce
    acc_a = xm.all_reduce(xm.REDUCE_SUM, buffer_a)
    acc_a = acc_a / W
    
    return acc_a