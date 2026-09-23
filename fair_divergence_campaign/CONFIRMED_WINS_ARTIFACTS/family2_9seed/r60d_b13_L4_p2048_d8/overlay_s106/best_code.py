def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 13
    OFF = 2
    
    # Step 1: Initial full all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute coverage counts for each block
    c = [0] * 13
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Perform 7 masked all-reduce iterations
    for iteration in range(7):
        # Create buffer with only kept blocks
        buf = torch.zeros_like(s)
        for b in range(13):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by coverage counts (skip on last iteration)
        if iteration < 6:
            for b in range(13):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s