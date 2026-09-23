def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    W = world_size
    L = 4
    
    # Precompute coverage counts for all blocks across all ranks
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        keep = set((start + 1*j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 rounds of selective all-reduces
    for round_idx in range(7):
        # Determine which blocks this rank keeps
        start = (rank + OFF) % B
        keep = set((start + 1*j) % B for j in range(L))
        
        # Create selective buffer
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the selective buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # For rounds 0-5, normalize by coverage counts
        # For round 6 (last round), skip normalization
        if round_idx < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s