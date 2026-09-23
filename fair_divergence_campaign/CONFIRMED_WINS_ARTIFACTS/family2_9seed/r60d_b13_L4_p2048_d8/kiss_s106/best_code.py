def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 13
    OFF = 2
    
    start = (rank + OFF) % B
    keep_set = set((start + j) % B for j in range(4))
    
    # Compute count for each bucket
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(4):
            c[(st + j) % B] += 1
    
    # Build both masks together
    combined_list = []
    for b in range(B):
        if b in keep_set:
            div_val = 1.0 / c[b] if c[b] > 0 else 1.0
            # Interleave weighted and regular mask values
            for _ in range(S):
                combined_list.append(div_val)
                combined_list.append(1.0)
        else:
            for _ in range(S):
                combined_list.append(0.0)
                combined_list.append(0.0)
    
    combined = torch.tensor(combined_list, device=x.device, dtype=x.dtype)
    # Split into two masks
    weighted_mask = combined[0::2]
    mask = combined[1::2]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Repeat pattern 6 times with division
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weighted_mask)
    
    # Last iteration without division
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s