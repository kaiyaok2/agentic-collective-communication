def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute scaling factors for each block
    c = [0] * 10
    for r in range(W):
        st = (r + 2) % 10
        ks = set((st + 1*j) % 10 for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Compute rank-dependent keep set once
    B = 10
    OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Batch the 7 iterations into fewer collective operations
    # Strategy: Do more local computation between collectives
    for iteration in range(7):
        # Create buffer with masked values (local op)
        buf = s.clone()  # Use clone instead of zeros + copy
        for b in range(10):
            if b not in keep:
                buf[b*S:(b+1)*S] = 0
        
        # All-reduce to sum the masked values
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by its precomputed factor (local op, vectorized)
        if iteration < 6:
            for b in range(10):
                acc[b*S:(b+1)*S] /= c[b]
        
        s = acc
    
    return s