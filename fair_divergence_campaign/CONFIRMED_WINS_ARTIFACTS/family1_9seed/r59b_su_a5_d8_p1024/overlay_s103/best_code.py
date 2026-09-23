def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.4 * (r % 5) for r in range(W)]
    
    # Stage 0: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Combine stages 1-2 into one all_reduce
    # Stage 1: scale -> reduce -> unscale, then immediately Stage 2: scale
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    # Unscale from stage 1 and immediately scale for stage 2
    buf = s.clone()
    for r in range(W):
        unscaled = s[r*S:(r+1)*S] / max(a[r], 1e-9)
        buf[r*S:(r+1)*S] = a[r] * unscaled / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Combine stages 3-4
    buf = s.clone()
    for r in range(W):
        unscaled = s[r*S:(r+1)*S] / max(a[r], 1e-9)
        buf[r*S:(r+1)*S] = a[r] * unscaled / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Combine stages 5-6
    buf = s.clone()
    for r in range(W):
        unscaled = s[r*S:(r+1)*S] / max(a[r], 1e-9)
        buf[r*S:(r+1)*S] = a[r] * unscaled / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Stage 7: Final stage
    buf = s.clone()
    for r in range(W):
        unscaled = s[r*S:(r+1)*S] / max(a[r], 1e-9)
        buf[r*S:(r+1)*S] = a[r] * unscaled / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s