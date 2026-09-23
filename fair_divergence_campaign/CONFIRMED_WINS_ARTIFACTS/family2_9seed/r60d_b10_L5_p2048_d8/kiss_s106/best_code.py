
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts
    c = [0] * 10
    for r in range(W):
        st = (r + 2) % 10
        ks = set((st + 1*j) % 10 for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Compute keep set for this rank
    B = 10
    OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Create mask once (reusable across iterations)
    mask = torch.zeros(10*S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create division factors once (avoid division by zero)
    div_factors = torch.ones(10*S, device=x.device, dtype=x.dtype)
    for b in range(10):
        if c[b] > 0:
            div_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Single iteration with division
    buf = s * mask
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc * div_factors
    
    # Final iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
