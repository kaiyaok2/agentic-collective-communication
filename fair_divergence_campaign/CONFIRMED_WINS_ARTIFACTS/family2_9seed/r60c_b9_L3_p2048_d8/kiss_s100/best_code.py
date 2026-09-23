def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 9
    OFF = 2
    
    # Precompute which buckets this rank keeps (as a list for iteration)
    start = (rank + OFF) % B
    keep = [(start + 1*j) % B for j in range(3)]
    
    # Precompute counts and create division factors tensor once
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(3):
            b = (st + 1*j) % B
            c[b] += 1
    
    # Create division tensor once
    div_factors = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with division
    for _ in range(6):
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * div_factors
    
    # Final iteration without division
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s