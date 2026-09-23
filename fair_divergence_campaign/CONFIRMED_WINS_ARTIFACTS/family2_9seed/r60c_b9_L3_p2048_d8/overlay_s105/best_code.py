def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial full all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute block counts
    c = [0] * 9
    for r in range(W):
        st = (r + 2) % 9
        ks = set((st + 1*j) % 9 for j in range(3))
        for b in ks:
            c[b] += 1
    
    # Perform 7 masked all-reduces with sequential scaling
    B = 9
    OFF = 2
    
    for iteration in range(7):
        # Compute keep set for this rank
        start = (rank + OFF) % B
        keep = set((start + 1*j) % B for j in range(3))
        
        # Create masked buffer
        buf = torch.zeros_like(s)
        for b in range(9):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by its count (except on last iteration)
        if iteration < 6:
            for b in range(9):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s