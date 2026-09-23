def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 12
    OFF = 2
    L = 5
    
    dtype = x.dtype
    
    # Pre-compute counts for each block (how many ranks cover it)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(L))
        for b in ks:
            c[b] += 1
    
    # Pre-compute this rank's keep set (same for all iterations)
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(L))
    
    # Stage 1: Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stage 2: Pipeline the 7 dependent mask-and-reduce operations
    # We group them into batches to reduce dispatches
    
    # First batch: iterations 0-2 (3 ops)
    for iteration in range(3):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Second batch: iterations 3-5 (3 ops)
    for iteration in range(3, 6):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 5:  # Skip division on last iteration
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    return s