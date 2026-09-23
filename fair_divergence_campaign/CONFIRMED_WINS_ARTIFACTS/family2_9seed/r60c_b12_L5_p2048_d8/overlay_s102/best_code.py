def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    L = 5
    W = world_size
    
    # Pre-compute which ranks cover each block
    block_to_ranks = [[] for _ in range(B)]
    for r in range(W):
        start = (r + OFF) % B
        for j in range(L):
            b = (start + j) % B
            block_to_ranks[b].append(r)
    
    # Compute coverage counts for normalization
    c = [len(block_to_ranks[b]) for b in range(B)]
    
    # Determine which blocks this rank covers
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Initial all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 dependent stages (8 total operations including initial all_reduce)
    for stage in range(7):
        # Create buffer with only blocks this rank covers
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the buffer (only non-zero where ranks cover blocks)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage count (except last stage)
        if stage < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s