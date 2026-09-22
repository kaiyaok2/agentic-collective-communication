def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute a[r] and perm[r]
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    # All-reduce 1: Initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # All-reduce 2: Apply permutation and scaling, then sum
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 3: Inverse scale, permute+scale, then sum
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 4: Inverse scale, permute+scale, then sum
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 5: Inverse scale, permute+scale, then sum
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 6: Inverse scale, permute+scale, then sum
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 7: Inverse scale, permute+scale, then sum
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # All-reduce 8: Inverse scale, permute+scale, then sum (final)
    for r in range(W):
        p = perm[r]
        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s