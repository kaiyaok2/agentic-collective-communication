def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts once
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(4):
            c[(st + j) % B] += 1
    
    # Create division factors tensor once (handle c[b]=0 case)
    div_factors = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
        else:
            div_factors[b*S:(b+1)*S] = 1.0
    
    # Create mask once
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(4))
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Apply 7 iterations
    for iteration in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:  # Divide on first 6 iterations
            s = acc * div_factors
        else:  # Last iteration: no division
            s = acc
    
    return s