def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute a and perm arrays
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    # First all-reduce: sum all inputs
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply permutation and scaling: buf[perm[r]] = a[r] * s[perm[r]] / W
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Second all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]: s[perm[r]] = s[perm[r]] / a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling again
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Third all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Fourth all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Fifth all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Sixth all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Seventh all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Divide by a[r]
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    
    # Apply permutation and scaling
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    
    # Eighth and final all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s