def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(3))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank should keep
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(3))
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process 7 dependent layers with optimized collective operations
    # Key optimization: combine masking operations to reduce local op overhead
    for layer in range(7):
        # Create mask once and reuse
        buf = s.clone()
        
        # Apply mask: zero out blocks we don't keep
        for b in range(B):
            if b not in keep:
                buf[b*S:(b+1)*S] = 0.0
        
        # All-reduce to sum the masked data
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by coverage count (except on last layer)
        # Vectorize the division operation
        if layer < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] /= c[b]
        
        s = acc
    
    return s