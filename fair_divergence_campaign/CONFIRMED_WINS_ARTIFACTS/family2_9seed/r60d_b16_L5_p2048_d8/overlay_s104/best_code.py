def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    L = 5
    W = world_size
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        keep = set((start + 1*j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Initial all-reduce to sum all input tensors
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 masked iterations
    for iteration in range(7):
        # Compute which blocks this rank keeps
        start = (rank + OFF) % B
        keep = set((start + 1*j) % B for j in range(L))
        
        # Create buffer with masked values (zero out blocks outside window)
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by coverage counts (except on last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s