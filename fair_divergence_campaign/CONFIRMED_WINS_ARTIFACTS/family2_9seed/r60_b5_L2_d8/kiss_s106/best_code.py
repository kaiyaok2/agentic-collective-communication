def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    B = 5
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute counts
    c = [0] * B
    for r in range(W):
        st = r % B
        ks = set((st + j) % B for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Create division factor tensor once
    div_factor = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        div_factor[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Determine which buckets this rank keeps and create mask once
    start = rank % B
    keep = set((start + j) % B for j in range(2))
    
    mask = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Perform 7 iterations
    for iteration in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by counts using vectorized multiplication (except last iteration)
        if iteration < 6:
            acc = acc * div_factor
        
        s = acc
    
    return s