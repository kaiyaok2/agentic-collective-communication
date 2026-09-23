def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 13
    OFF = 2
    L = 4
    
    # First all-reduce: sum all x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute counts: how many ranks cover each block
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        keep = set((start + 1 * j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1 * j) % B for j in range(L))
    
    # Perform 7 masked all-reduces with division
    for iteration in range(7):
        # Mask: zero out blocks not in keep
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by counts (skip on last iteration as per reference)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s