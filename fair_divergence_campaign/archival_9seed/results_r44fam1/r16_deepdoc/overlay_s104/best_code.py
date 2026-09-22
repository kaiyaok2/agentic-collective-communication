def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Buffer multiple stages before synchronizing
    # We'll do stages in groups: (2,3), (4,5), (6,7), (8)
    # Each pair can be prepared together, then synchronized
    
    # Stages 2-3 (buffered)
    buf2 = s.clone()
    for r in range(W):
        buf2[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    
    for r in range(W):
        s2[r*S:(r+1)*S] = s2[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf3 = s2.clone()
    for r in range(W):
        buf3[r*S:(r+1)*S] = a[r] * s2[r*S:(r+1)*S] / W
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    
    # Stages 4-5 (buffered)
    for r in range(W):
        s3[r*S:(r+1)*S] = s3[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf4 = s3.clone()
    for r in range(W):
        buf4[r*S:(r+1)*S] = a[r] * s3[r*S:(r+1)*S] / W
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    
    for r in range(W):
        s4[r*S:(r+1)*S] = s4[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf5 = s4.clone()
    for r in range(W):
        buf5[r*S:(r+1)*S] = a[r] * s4[r*S:(r+1)*S] / W
    s5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    
    # Stages 6-7 (buffered)
    for r in range(W):
        s5[r*S:(r+1)*S] = s5[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf6 = s5.clone()
    for r in range(W):
        buf6[r*S:(r+1)*S] = a[r] * s5[r*S:(r+1)*S] / W
    s6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    
    for r in range(W):
        s6[r*S:(r+1)*S] = s6[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf7 = s6.clone()
    for r in range(W):
        buf7[r*S:(r+1)*S] = a[r] * s6[r*S:(r+1)*S] / W
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf7)
    
    # Stage 8 (final)
    for r in range(W):
        s7[r*S:(r+1)*S] = s7[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf8 = s7.clone()
    for r in range(W):
        buf8[r*S:(r+1)*S] = a[r] * s7[r*S:(r+1)*S] / W
    s8 = xm.all_reduce(xm.REDUCE_SUM, buf8)
    
    return s8