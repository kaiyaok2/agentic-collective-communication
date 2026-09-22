
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Create mask tensor once (vectorized)
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization tensor once (vectorized, avoid division by zero)
    norm = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterative masked all-reduce with normalization
    for iteration in range(7):
        # Apply mask via element-wise multiplication
        buf = s * mask
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize (except last iteration)
        if iteration < 6:
            s = s * norm
    
    return s
