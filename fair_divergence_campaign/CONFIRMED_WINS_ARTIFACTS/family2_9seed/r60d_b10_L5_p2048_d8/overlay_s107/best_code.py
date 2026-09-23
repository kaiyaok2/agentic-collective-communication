def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    
    # Precompute the normalization factors for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Initial all-reduce to sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Perform 7 iterations of the dependent all-reduce pattern
    # Each iteration: mask blocks, all-reduce, normalize
    for iteration in range(7):
        # Create buffer and mask out blocks not in keep set
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize each block by its count (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s