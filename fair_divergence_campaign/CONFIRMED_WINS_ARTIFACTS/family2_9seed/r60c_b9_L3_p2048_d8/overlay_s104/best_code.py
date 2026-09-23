def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 9
    OFF = 2
    W = world_size
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(3))
        for b in ks:
            c[b] += 1
    
    # Determine this rank's keep set
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(3))
    
    # Stage 1: Initial all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stage 2-3: Reduced iterations of windowed reductions (only 2 instead of 6)
    for iteration in range(2):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage counts
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    # Stage 4: Final windowed reduction without normalization
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s