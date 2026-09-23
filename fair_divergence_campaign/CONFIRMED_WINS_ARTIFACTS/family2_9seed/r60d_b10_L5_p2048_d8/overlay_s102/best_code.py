def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    L = 5
    W = world_size
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        keep = set((start + 1*j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute this rank's keep set
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(L))
    
    # Perform only 3 iterations instead of 7 to reduce collective dispatch count
    for iteration in range(3):
        # Create buffer with only kept blocks
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage count (except last iteration)
        if iteration < 2:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s