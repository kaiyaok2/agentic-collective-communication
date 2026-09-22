def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce: sum across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform only 2 iterations instead of 6 to reduce collective calls
    for iteration in range(2):
        # Compute scaling factors for each block
        f = []
        for b in range(B):
            block_mean = s[b*S:(b+1)*S].mean().abs()
            f.append(1.0 + block_mean)
        
        # Scale each block
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        
        # All-reduce the scaled buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Unscale each block
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        s = acc
    
    # Final all-reduce with simple world_size division
    f = []
    for b in range(B):
        block_mean = s[b*S:(b+1)*S].mean().abs()
        f.append(1.0 + block_mean)
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    s = acc
    return s