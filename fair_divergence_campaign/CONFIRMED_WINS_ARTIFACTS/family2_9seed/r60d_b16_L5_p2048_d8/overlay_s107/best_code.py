def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    L = 5
    W = world_size
    dtype = x.dtype
    
    # Compute count array: how many ranks contribute to each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(L))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank owns
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(L))
    
    # Phase 1: Reduce-scatter using block-wise masking
    # Each rank reduces only its owned blocks across all ranks
    # We'll do this in multiple rounds to simulate reduce-scatter behavior
    
    # Step 1: Initial all-reduce to get sum of all data
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Perform 7 rounds of selective reduction (8 dependent ops total including first)
    for iteration in range(7):
        # Each rank zeros out blocks it doesn't own
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce to sum up owned blocks
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by count (except on last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s