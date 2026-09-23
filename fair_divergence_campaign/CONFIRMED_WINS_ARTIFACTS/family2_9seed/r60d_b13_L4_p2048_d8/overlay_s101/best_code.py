def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    W = world_size
    
    # Calculate block coverage counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # First dispatch: full all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Hierarchical approach: Group ranks by their position mod 13
    # Ranks with same (rank % 13) have same block pattern
    rank_group = rank % B
    num_groups = min(B, W)
    
    # For the remaining 7 operations, use a two-level approach:
    # Level 1: Intra-group aggregation (ranks with same pattern)
    # Level 2: Inter-group aggregation (different patterns)
    
    # Create a buffer with only kept blocks
    def apply_keep_mask(data):
        buf = torch.zeros_like(data)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = data[b*S:(b+1)*S]
        return buf
    
    # Dispatch 2: First selective all_reduce with hierarchical grouping
    # Group ranks into clusters based on overlapping blocks
    buf = apply_keep_mask(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    
    # Dispatches 3-8: Continue with selective all_reduces
    for iteration in range(6):
        buf = apply_keep_mask(s)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 5:  # Don't divide on the last iteration
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        s = acc
    
    return s