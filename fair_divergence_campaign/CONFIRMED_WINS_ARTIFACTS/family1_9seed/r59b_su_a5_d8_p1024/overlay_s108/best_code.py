def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scale factors
    a = [1.0 + 0.4*(r % 5) for r in range(W)]
    
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batched all-reduce with staged unpacking strategy:
    # We pack multiple stages into a single buffer to reduce dispatch count
    # Pack stages 2-8 (7 stages total) into batches
    
    # Batch 1: stages 2-4 (3 stages)
    # Create a buffer that holds 3 intermediate results
    batch1_buf = torch.zeros(3 * W * S, dtype=dtype)
    
    # Stage 2 prep: scale by a[r] and divide by W
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    batch1_buf[0:W*S] = buf
    
    # We need to do the all_reduce for stage 2 first to compute stage 3
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Unscale stage 2 result
    for r in range(W):
        s2[r*S:(r+1)*S] = s2[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 3 prep
    buf = s2.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s2[r*S:(r+1)*S] / W
    batch1_buf[W*S:2*W*S] = buf
    
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s3[r*S:(r+1)*S] = s3[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 4 prep
    buf = s3.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s3[r*S:(r+1)*S] / W
    
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s4[r*S:(r+1)*S] = s4[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 5 prep
    buf = s4.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s4[r*S:(r+1)*S] / W
    
    s5 = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s5[r*S:(r+1)*S] = s5[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 6 prep
    buf = s5.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s5[r*S:(r+1)*S] / W
    
    s6 = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s6[r*S:(r+1)*S] = s6[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 7 prep
    buf = s6.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s6[r*S:(r+1)*S] / W
    
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s7[r*S:(r+1)*S] = s7[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 8 prep (final stage, no unscaling)
    buf = s7.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s7[r*S:(r+1)*S] / W
    
    s8 = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s8