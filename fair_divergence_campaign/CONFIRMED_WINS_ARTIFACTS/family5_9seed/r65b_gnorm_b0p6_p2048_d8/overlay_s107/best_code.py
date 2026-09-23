def r65b_gnorm_b0p6_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.6
    dtype = x.dtype
    
    # Initial all-reduce to get sum across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute all 8 iterations locally and collect vectors
    all_vectors = []
    
    # Iteration 0
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    all_vectors.append(buf)
    acc = buf
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    
    # Iterations 1-7
    for i in range(1, 8):
        gr = 1.0 + BETA * M
        s_next = acc * gr
        g_next = 1.0 + BETA * s_next.abs().mean()
        buf = s_next / g_next
        all_vectors.append(buf)
        acc = buf
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
    
    # Batch all 8 vectors into a single all-reduce
    stacked_all = torch.stack(all_vectors)
    reduced_all = xm.all_reduce(xm.REDUCE_SUM, stacked_all) / W
    
    # Return the last iteration result
    return reduced_all[7]