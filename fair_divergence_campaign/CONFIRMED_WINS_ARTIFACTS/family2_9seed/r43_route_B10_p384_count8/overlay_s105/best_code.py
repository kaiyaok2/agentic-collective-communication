def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Precompute the keep-set once (rank-indexed routing: THIS rank masks its own length-L window)
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Loop over 7 mask-allreduce-normalize triplets with in-place updates
    for iteration in range(7):
        # Mask: keep only blocks in the window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by overlap count (skip on last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b * S:(b + 1) * S] = acc[b * S:(b + 1) * S] / c[b]
        
        s = acc
    
    return s