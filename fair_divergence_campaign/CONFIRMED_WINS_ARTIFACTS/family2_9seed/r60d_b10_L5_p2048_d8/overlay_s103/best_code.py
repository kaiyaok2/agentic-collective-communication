def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    L = 5
    W = world_size
    
    # Pre-compute count array: how many ranks' windows cover each block
    c = [0] * B
    for r in range(W):
        start = (r + OFF) % B
        keep = set((start + 1 * j) % B for j in range(L))
        for b in keep:
            c[b] += 1
    
    # Initial full all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute this rank's window
    start = (rank + OFF) % B
    keep = set((start + 1 * j) % B for j in range(L))
    
    # Reduced to 2 masked window all-reduce operations
    for iteration in range(2):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by counts (skip division on last iteration)
        if iteration < 1:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s