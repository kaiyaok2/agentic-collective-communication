def r65b_gnorm_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    dtype = x.dtype
    
    # First all-reduce: get initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process iterations 0-5 locally with batched all-reduce
    buffers = []
    
    # Iteration 0
    g0 = 1.0 + BETA * s.abs().mean()
    buf0 = s / g0
    buffers.append(buf0)
    
    # Iterations 1-5: speculate based on local data
    current = buf0
    for i in range(1, 6):
        A = current.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s_next = current * gr
        g_next = 1.0 + BETA * s_next.abs().mean()
        buf_next = s_next / g_next
        buffers.append(buf_next)
        current = buf_next
    
    # Single batched all-reduce for all iterations 0-5
    combined = torch.cat(buffers)
    combined_reduced = xm.all_reduce(xm.REDUCE_SUM, combined)
    
    # Extract iteration 5 result and compute iteration 6
    size = buffers[0].numel()
    acc5 = combined_reduced[5*size:6*size] / W
    
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s6 = acc5 * gr5
    g6 = 1.0 + BETA * s6.abs().mean()
    buf6 = s6 / g6
    
    # Final all-reduce for iteration 6
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    
    return acc6