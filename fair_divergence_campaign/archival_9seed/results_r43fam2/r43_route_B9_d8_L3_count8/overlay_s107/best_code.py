def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Direct Sequential Masked All-Reduce Chain
    
    Performs 8 sequential all-reduce operations:
    1. Initial sum across all ranks
    2-8. Seven masked iterations with per-block scaling
    
    Each iteration:
    - Masks to keep only the length-3 window for this rank
    - All-reduce sum
    - Scale each block by the number of ranks whose window covers it
    """
    S = 256
    B = 9
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Compute per-block overlap count: c[b] = number of ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Steps 2-8: Seven masked iterations
    for iteration in range(7):
        # Compute this rank's window
        start = (rank + OFF) % B
        keep = set((start + STR * j) % B for j in range(L))
        
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by overlap count (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s