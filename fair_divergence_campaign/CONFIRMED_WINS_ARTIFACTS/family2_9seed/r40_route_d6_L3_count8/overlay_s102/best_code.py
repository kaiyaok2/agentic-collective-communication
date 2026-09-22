def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Pre-compute per-block overlap counts (constant across all ranks)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Pre-compute THIS rank's window mask (constant across all iterations)
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused loop for 6 dependent all_reduce operations
    # Iterations 0-4: mask + all_reduce + scale
    # Iteration 5: mask + all_reduce (no scale needed)
    for iteration in range(6):
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale (skip on last iteration)
        if iteration < 5:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s