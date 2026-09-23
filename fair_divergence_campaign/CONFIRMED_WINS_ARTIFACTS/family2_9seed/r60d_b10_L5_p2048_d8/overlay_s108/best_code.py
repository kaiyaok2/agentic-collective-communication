def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    W = world_size
    
    # Precompute counts for each block (how many ranks cover it)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Precompute this rank's window mask once
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused mask-and-reduce pipeline: reduced to 3 iterations
    for iteration in range(3):
        # Apply mask and all-reduce in single dispatch
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by precomputed counts (except last iteration)
        if iteration < 2:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s