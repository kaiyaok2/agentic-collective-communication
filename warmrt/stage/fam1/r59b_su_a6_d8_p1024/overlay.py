def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.35*(r % 6) for r in range(W)]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process stages 2-8 in batches
    # We batch stages to reduce synchronization overhead
    # Strategy: Prepare multiple buffers and try to pipeline operations
    
    # Stage 1: Already done (initial all_reduce)
    
    # Batch stages 2-3
    buf1 = s.clone()
    for r in range(W):
        buf1[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    for r in range(W):
        s1[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf2 = s1.clone()
    for r in range(W):
        buf2[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    
    # Batch stages 4-5
    for r in range(W):
        s2[r*S:(r+1)*S] = s2[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf3 = s2.clone()
    for r in range(W):
        buf3[r*S:(r+1)*S] = a[r] * s2[r*S:(r+1)*S] / W
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    
    for r in range(W):
        s3[r*S:(r+1)*S] = s3[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf4 = s3.clone()
    for r in range(W):
        buf4[r*S:(r+1)*S] = a[r] * s3[r*S:(r+1)*S] / W
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    
    # Batch stages 6-7
    for r in range(W):
        s4[r*S:(r+1)*S] = s4[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf5 = s4.clone()
    for r in range(W):
        buf5[r*S:(r+1)*S] = a[r] * s4[r*S:(r+1)*S] / W
    s5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    
    for r in range(W):
        s5[r*S:(r+1)*S] = s5[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf6 = s5.clone()
    for r in range(W):
        buf6[r*S:(r+1)*S] = a[r] * s5[r*S:(r+1)*S] / W
    s6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    
    # Stage 8
    for r in range(W):
        s6[r*S:(r+1)*S] = s6[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf7 = s6.clone()
    for r in range(W):
        buf7[r*S:(r+1)*S] = a[r] * s6[r*S:(r+1)*S] / W
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf7)
    
    return s7