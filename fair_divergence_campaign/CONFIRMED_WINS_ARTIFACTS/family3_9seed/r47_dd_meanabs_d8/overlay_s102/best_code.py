def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce: sum input x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute all 7 scaling factors after first all-reduce
    # (We'll compute them iteratively as we refine)
    
    # Iteration 1
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    # Apply scaling: buf = s * f (per block)
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Unscale: divide by (world_size * f)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 2
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 3
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 3
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 4
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 4
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 5
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 5
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 6
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 7
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 7 (final)
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    # All-reduce 8 (final)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc