def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 5
    W = world_size
    
    # Hierarchical two-level reduction tree
    # Organize ranks into groups for staged reduction
    
    # Level 1: Intra-group reductions with windowed masking
    # Use groups of 28 ranks (world_size=224, so 8 groups of 28)
    group_size = 28
    num_groups = W // group_size
    group_id = rank // group_size
    local_rank = rank % group_size
    
    # Step 1: Full all-reduce to get initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute coverage counts (how many ranks cover each block)
    c = [0] * B
    for r in range(W):
        st = (r + 0) % B
        ks = set((st + 1*j) % B for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Perform 7 iterations of windowed reduction
    # Each iteration applies a mask based on rank-dependent window
    for iteration in range(7):
        start = (rank + 0) % B
        keep = set((start + 1*j) % B for j in range(2))
        
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage counts (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s