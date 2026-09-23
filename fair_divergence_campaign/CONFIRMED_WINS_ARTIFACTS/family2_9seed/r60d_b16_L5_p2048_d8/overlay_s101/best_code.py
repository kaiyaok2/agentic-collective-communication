def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # First all-reduce: sum all ranks' input
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute counts for each block
    c = [0] * 16
    for r in range(W):
        st = (r + 2) % 16
        ks = set((st + 1*j) % 16 for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank keeps
    B = 16
    OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Perform 7 iterations of mask-reduce-average
    for iteration in range(7):
        # Create buffer with selective masking
        buf = torch.zeros_like(s)
        for b in range(16):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Average each block by its count (skip on last iteration)
        if iteration < 6:
            for b in range(16):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s