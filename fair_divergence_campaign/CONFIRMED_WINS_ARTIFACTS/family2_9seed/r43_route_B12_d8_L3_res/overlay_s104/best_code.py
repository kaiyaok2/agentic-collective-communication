def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap count c[b] = number of ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank's window covers
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Reduce iterations by doing more work per iteration
    # Instead of 7 iterations, do 3 iterations with combined operations
    for iteration in range(3):
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce to sum masked values
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap count
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    # Final masking and all-reduce without scaling
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s