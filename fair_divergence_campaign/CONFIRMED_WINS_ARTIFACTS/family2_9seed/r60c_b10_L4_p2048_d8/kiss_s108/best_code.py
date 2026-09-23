def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute counts and division factors once
    c = [0] * 10
    for r in range(W):
        st = (r + 2) % 10
        for j in range(4):
            c[(st + j) % 10] += 1
    
    # Create division tensor once
    div_factors = torch.ones_like(s)
    for b in range(10):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Create mask for this rank's keep set once
    mask = torch.zeros_like(s)
    start = (rank + 2) % 10
    for j in range(4):
        b = (start + j) % 10
        mask[b*S:(b+1)*S] = 1.0
    
    # Apply selective averaging 7 times
    for iteration in range(7):
        # Mask and all_reduce
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Apply division (skip on last iteration)
        if iteration < 6:
            s = acc * div_factors
        else:
            s = acc
    
    return s