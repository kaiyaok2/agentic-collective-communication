def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Step 1: Initial all_reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Pre-compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Step 3: Pre-compute which blocks THIS rank contributes to across all 7 rounds
    # Each rank's window is the same in all rounds: length-L window starting at (rank+OFF) % B
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Step 4: Fused mask-aggregate-normalize for rounds 1-6 (round 7 has no final normalization)
    # We'll do 6 rounds with normalization, then 1 final round without final normalization
    
    for round_idx in range(6):
        # Mask: THIS rank keeps only its window blocks
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # Aggregate via all_reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by per-block overlap count
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    # Step 5: Final (7th) round - mask and aggregate, but no normalization at the end
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc
    
    return s