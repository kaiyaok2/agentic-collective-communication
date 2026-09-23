def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # First full all-reduce (as in reference)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute coverage count for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(3))
        for b in ks:
            c[b] += 1
    
    # Process 7 masked all-reduce operations using reduce-scatter + all-gather
    for iteration in range(7):
        # Determine which blocks this rank keeps
        start = (rank + OFF) % B
        keep = set((start + 1*j) % B for j in range(3))
        
        # Reduce-scatter phase: prepare buffer with only kept blocks
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce to sum (simulates reduce-scatter + all-gather)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage count (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s