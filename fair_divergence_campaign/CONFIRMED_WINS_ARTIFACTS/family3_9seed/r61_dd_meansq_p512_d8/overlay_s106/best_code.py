def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create two buffers for pipelining
    buf0 = s.clone()
    buf1 = s.clone()
    
    # Process 8 iterations in pairs using dual buffers
    # Iteration 0 (uses buf0)
    f0 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f0.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf0[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f0[b]
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Iteration 1 (uses buf1, overlaps with acc0 finalization)
    for b in range(B):
        acc0[b*S:(b+1)*S] = acc0[b*S:(b+1)*S] / (world_size * f0[b])
    s = acc0
    f1 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f1.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf1[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f1[b]
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Iteration 2 (uses buf0, overlaps with acc1 finalization)
    for b in range(B):
        acc1[b*S:(b+1)*S] = acc1[b*S:(b+1)*S] / (world_size * f1[b])
    s = acc1
    f2 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f2.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf0[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f2[b]
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Iteration 3 (uses buf1, overlaps with acc0 finalization)
    for b in range(B):
        acc0[b*S:(b+1)*S] = acc0[b*S:(b+1)*S] / (world_size * f2[b])
    s = acc0
    f3 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f3.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf1[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f3[b]
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Iteration 4 (uses buf0, overlaps with acc1 finalization)
    for b in range(B):
        acc1[b*S:(b+1)*S] = acc1[b*S:(b+1)*S] / (world_size * f3[b])
    s = acc1
    f4 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f4.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf0[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f4[b]
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Iteration 5 (uses buf1, overlaps with acc0 finalization)
    for b in range(B):
        acc0[b*S:(b+1)*S] = acc0[b*S:(b+1)*S] / (world_size * f4[b])
    s = acc0
    f5 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f5.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf1[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f5[b]
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Iteration 6 (uses buf0, overlaps with acc1 finalization)
    for b in range(B):
        acc1[b*S:(b+1)*S] = acc1[b*S:(b+1)*S] / (world_size * f5[b])
    s = acc1
    f6 = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f6.append(1.0 + (sb*sb).mean())
    for b in range(B):
        buf0[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f6[b]
    acc0 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Final iteration (iteration 7) - no normalization by f, just world_size
    acc0 = acc0 / world_size
    s = acc0
    
    return s