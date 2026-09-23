def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(3):
            c[(st + 1*j) % B] += 1
    
    # Precompute keep indices
    start = (rank + OFF) % B
    keep = [(start + 1*j) % B for j in range(3)]
    
    # Create selection mask once
    select_mask = torch.zeros_like(s)
    for b in keep:
        select_mask[b*S:(b+1)*S] = 1
    
    # Create normalization tensor once
    norm = torch.ones_like(s)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Perform 6 iterations with normalization
    for _ in range(6):
        buf = s * select_mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * norm
    
    # Final iteration without normalization
    buf = s * select_mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s