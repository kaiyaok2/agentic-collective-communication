def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 9
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    dtype = x.dtype
    
    # Precompute overlap counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Precompute this rank's window mask once
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Step 1: Initial all_reduce on full input
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Steps 2-8: 7 masked all_reduce operations
    # Each iteration: mask -> all_reduce -> scale by overlap counts
    for iteration in range(7):
        # Create masked buffer (only keeping this rank's window blocks)
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by its overlap count
        # (except on the last iteration where we skip scaling)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s