def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    dtype = x.dtype
    
    # Compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Step 1: Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Apply local windowing for all 5 intermediate levels
    # Each level applies windowing and scaling in a single pass
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Process all 5 levels with local operations only
    for level in range(5):
        # Apply windowing: zero out blocks not in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] / c[b]
        
        # All-reduce to sum the masked and scaled contributions
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final level (6th): apply windowing and sum
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s