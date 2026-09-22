def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Compute per-block overlap count
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Initial all-reduce to get sum of x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Split into chunks for pipelined processing
    # We'll use 4 chunks of 3 blocks each to balance dispatch count and overlap
    chunk_size = 3  # blocks per chunk
    num_chunks = 4
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Process 7 iterations with pipelined async operations
    for iteration in range(7):
        # Create buffer for this iteration
        buf = torch.zeros_like(s)
        
        # Mask: only keep blocks in this rank's window
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap count (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s