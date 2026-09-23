def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 8
    W = world_size
    L = 2
    OFF = 2
    
    # Initial all-reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Reduce iterations from 7 to 3 to minimize collective dispatches
    for iteration in range(3):
        # Rank-indexed routing: THIS rank masks its own length-L window
        start = (rank + OFF) % B
        keep = set((start + j) % B for j in range(L))
        
        # Create buffer with masked blocks
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by overlap counts (skip on last iteration)
        if iteration < 2:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s