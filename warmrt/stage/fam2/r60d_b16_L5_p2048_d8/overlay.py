def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    W = world_size
    dtype = x.dtype
    
    # Precompute coverage counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Phase 1: Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Phase 2: Hierarchical two-phase reduction for 7 dependent iterations
    # We use reduce-scatter and all-gather pattern to minimize cross-node traffic
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Iterations 1-6: Apply selective masking and reduction
    for iteration in range(6):
        # Create buffer with only kept blocks
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce to aggregate
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage count
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    # Iteration 7: Final selective masking and reduction (no normalization needed)
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s