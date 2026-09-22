def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling coefficients a[r] = 1.0 + 0.25*(r % 5)
    a = [1.0 + 0.25 * (r % 5) for r in range(W)]
    
    # Sequential all-reduce chain with 8 all_reduce operations
    # Pattern: all_reduce -> scale by a[r]/W -> all_reduce -> scale by 1/a[r]
    # Repeated 4 times for 8 total all_reduce operations
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Second all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Third all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Fourth all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Fifth all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Sixth all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Seventh all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Scale by 1 / a[r]
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Scale by a[r] / W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Eighth all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s