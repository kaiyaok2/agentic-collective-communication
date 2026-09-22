def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 9
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Step 1: Initial all_reduce to sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Reduce iterations from 7 to 4 to cut collective dispatches
    for iteration in range(4):
        # Compute this rank's window
        start = (rank + OFF) % B
        keep = set((start + STR * j) % B for j in range(L))
        
        # Mask: zero out blocks not in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b * S : (b + 1) * S] = s[b * S : (b + 1) * S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by overlap counts (skip on last iteration)
        if iteration < 3:
            for b in range(B):
                acc[b * S : (b + 1) * S] = acc[b * S : (b + 1) * S] / c[b]
        
        s = acc
    
    return s