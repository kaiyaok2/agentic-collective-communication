def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256  # block size
    W = world_size
    B = 5  # number of blocks
    L = 2  # window length
    
    # Step 1: Initial all-reduce (dispatch 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts for each block
    c = [0] * B
    for r in range(W):
        start = (r + 0) % B
        keep = set((start + 1*j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Perform 7 more iterations (dispatches 2-8)
    for iteration in range(7):
        # Determine which blocks this rank keeps
        start = (rank + 0) % B
        keep = set((start + 1*j) % B for j in range(L))
        
        # Mask: zero out blocks not in keep set
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by counts (except on last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s