
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 12
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Pre-compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Compute which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Create normalization factor tensor once  
    norm_factors = torch.ones(B * S, dtype=x.dtype, device=x.device)
    for b in range(B):
        if c[b] > 0:
            norm_factors[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 7 iterations of mask-reduce-normalize
    for iteration in range(7):
        # Create masked buffer
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All_reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by overlap counts (except last iteration)
        if iteration < 6:
            acc = acc * norm_factors
        
        s = acc
    
    return s
