def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    B = 5
    dtype = x.dtype
    
    # Compute coverage counts for each block
    c = torch.tensor([0.0] * B, dtype=dtype, device=x.device)
    for r in range(W):
        st = (r + 0) % B
        ks = set((st + 1*j) % B for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which blocks this rank should keep
    OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    
    # Create windowed version once
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Single all-reduce for windowed sum
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Normalize by coverage and apply iteratively (all local ops)
    result = s.clone()
    for iteration in range(7):
        for b in range(B):
            if iteration < 6:
                result[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
            else:
                result[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    
    return result