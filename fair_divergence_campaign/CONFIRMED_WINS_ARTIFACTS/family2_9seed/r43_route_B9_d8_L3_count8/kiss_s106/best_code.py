def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = set((start + STR*j) % B for j in range(L))
    
    # Create mask tensor once (1 for kept blocks, 0 otherwise)
    mask_list = []
    for b in range(B):
        if b in keep_blocks:
            mask_list.extend([1.0] * S)
        else:
            mask_list.extend([0.0] * S)
    mask = torch.tensor(mask_list, device=x.device, dtype=x.dtype)
    
    # Create division factors once (avoid division by zero)
    div_list = []
    for b in range(B):
        if c[b] > 0:
            div_list.extend([1.0 / c[b]] * S)
        else:
            div_list.extend([1.0] * S)
    div_factors = torch.tensor(div_list, device=x.device, dtype=x.dtype)
    
    # 6 iterations with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * div_factors
    
    # Final iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s