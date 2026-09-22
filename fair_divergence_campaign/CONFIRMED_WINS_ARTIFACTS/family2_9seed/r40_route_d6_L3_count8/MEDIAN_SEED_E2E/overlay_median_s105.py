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
    
    # Phase 1: Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine this rank's window
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Hierarchical approach: exploit structure by doing node-local aggregation
    # With world_size=224, we can assume some node structure (e.g., 28 nodes × 8 ranks/node)
    # However, since we don't have explicit node info, we'll optimize by batching operations
    
    # Phase 2: Five stages of mask-aggregate-scale
    # We can potentially fuse some operations, but must maintain correctness
    
    for stage in range(5):
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # Aggregate across all ranks
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by overlap counts (except last stage - see reference)
        if stage < 4:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s