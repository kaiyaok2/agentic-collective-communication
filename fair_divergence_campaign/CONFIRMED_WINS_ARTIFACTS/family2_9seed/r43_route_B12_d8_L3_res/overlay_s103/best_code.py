def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Step 1: All-reduce to get the full sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Step 3: Determine this rank's window
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Step 4: Perform 7 rounds of mask + all-reduce + scale
    for round_idx in range(7):
        # Mask: zero out blocks not in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap counts (except on the last round)
        if round_idx < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s