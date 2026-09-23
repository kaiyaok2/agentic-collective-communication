def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.4*(r % 5) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse stages 2-8 into fewer collective operations
    # Each iteration does 2 stages worth of work before the next all-reduce
    
    # Stages 2-3 fused
    buf = s.clone()
    for r in range(W):
        # Stage 2 transform
        temp = a[r] * s[r*S:(r+1)*S] / W
        buf[r*S:(r+1)*S] = temp
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    # Stage 3 local ops (no all-reduce yet)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stages 4-5 fused
    buf = s.clone()
    for r in range(W):
        temp = a[r] * s[r*S:(r+1)*S] / W
        buf[r*S:(r+1)*S] = temp
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stages 6-7 fused
    buf = s.clone()
    for r in range(W):
        temp = a[r] * s[r*S:(r+1)*S] / W
        buf[r*S:(r+1)*S] = temp
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 8
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s