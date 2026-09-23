def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    L = 4
    
    # Compute coverage counts (how many ranks cover each block)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(L))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(L))
    
    # Initial all-reduce (dispatch 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reduced iterations: 2-5 (4 iterations instead of 6)
    for iteration in range(4):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Final iteration: mask and all-reduce (no normalization)
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s