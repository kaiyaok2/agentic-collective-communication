def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Initial global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine this rank's window
    start = (rank + OFF) % B
    my_window = set((start + j) % B for j in range(L))
    
    # Group ranks into cohorts based on their window
    # Since B=8, L=3, OFF=2, there are 8 possible windows (one per rank if W=8)
    # But the key insight: ranks with the same window share identical keep-masks
    # With W=8 ranks and B=8 blocks, each rank has a unique window
    
    # However, we can still exploit structure: 
    # Instead of doing 6 separate masked allreduces, we aggregate hierarchically
    
    # Strategy: Split the 6 iterations into two phases of 3 iterations each
    # Phase 1: 3 iterations
    for iteration in range(3):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in my_window:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Phase 2: 3 iterations (but last one doesn't need division)
    for iteration in range(2):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in my_window:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    # Final iteration (no division needed as per reference)
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in my_window:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s