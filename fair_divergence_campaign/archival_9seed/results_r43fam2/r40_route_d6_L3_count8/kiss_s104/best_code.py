def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Avoid division by zero
    c_safe = [max(1, cnt) for cnt in c]
    
    # Create combined mask_div tensor that includes both masking and division
    # For efficiency, create it in one pass
    start = (rank + OFF) % B
    keep_set = set((start + j) % B for j in range(L))
    
    mask = torch.zeros_like(s)
    div_factors = torch.zeros_like(s)
    
    for b in range(B):
        if b in keep_set:
            mask[b*S:(b+1)*S] = 1.0
        div_factors[b*S:(b+1)*S] = 1.0 / c_safe[b]
    
    # Two iterations
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask) * div_factors
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s