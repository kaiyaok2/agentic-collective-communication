def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 2
    
    # Step 1: Initial all_reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap count c[b] = number of ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine THIS rank's window once
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Perform only 2 iterations to minimize collective dispatches
    for iteration in range(2):
        # Mask: zero out blocks not in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by overlap counts (skip on last iteration)
        if iteration < 1:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s