def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors a[r] = 1.0 + 0.4*(r % 5)
    a = [1.0 + 0.4 * (r % 5) for r in range(W)]
    
    # Phase 1: Batch stages 1-4
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-4: Apply scale, reduce, unscale pattern 3 times
    for stage_idx in range(3):
        buf = s.clone()
        for r in range(W):
            buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Phase 2: Batch stages 5-8
    # Stages 5-8: Apply scale, reduce, unscale pattern 4 times
    for stage_idx in range(4):
        buf = s.clone()
        for r in range(W):
            buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Only unscale if not the last stage
        if stage_idx < 3:
            for r in range(W):
                s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    return s