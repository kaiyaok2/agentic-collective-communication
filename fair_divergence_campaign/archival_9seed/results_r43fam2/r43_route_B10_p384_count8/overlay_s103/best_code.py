def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute this rank's window blocks
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Pipeline: Issue all 7 all-reduce operations with interleaved computation
    # We have 7 iterations (6 mask+divide+all_reduce, then 1 final mask+all_reduce)
    
    for iteration in range(6):
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce (asynchronously dispatched by XLA)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by overlap counts
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    # Final iteration (7th): mask and all-reduce without division
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s