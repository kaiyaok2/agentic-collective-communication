def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 16
    OFF = 2
    
    # Compute coverage counts for each block
    c = [0] * 16
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Get this rank's keep set
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process 7 windowed all-reduces with optimized collectives
    for iteration in range(7):
        # Mask out blocks not in keep set (local op)
        buf = s.clone()
        for b in range(B):
            if b not in keep:
                buf[b*S:(b+1)*S] = 0
        
        # Single all-reduce combines reduce-scatter and all-gather
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage counts (local op, except last iteration)
        if iteration < 6:
            for b in range(B):
                if c[b] > 0:
                    s[b*S:(b+1)*S] = s[b*S:(b+1)*S] / c[b]
    
    return s