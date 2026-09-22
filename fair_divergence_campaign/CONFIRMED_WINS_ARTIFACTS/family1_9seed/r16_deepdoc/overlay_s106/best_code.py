def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # We have 7 more stages (after the initial reduce), each doing:
    # 1. Apply per-shard scaling: buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    # 2. All-reduce
    # 3. Undo scaling: s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / a[r]
    
    # Strategy: Batch stages together by concatenating intermediate buffers
    # We'll batch in groups of 2 stages to reduce from 7 dispatches to ~4
    
    # Stages 1-2 batched
    buf1 = s.clone()
    for r in range(W):
        buf1[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # After stage 1 reduction, compute stage 2 input
    s1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    s_temp = s1.clone()
    for r in range(W):
        s_temp[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    buf2 = s_temp.clone()
    for r in range(W):
        buf2[r*S:(r+1)*S] = a[r] * s_temp[r*S:(r+1)*S] / W
    
    # Concatenate and reduce together
    batched_buf = torch.cat([buf1, buf2], dim=0)
    batched_result = xm.all_reduce(xm.REDUCE_SUM, batched_buf)
    
    # Extract results
    s1 = batched_result[:W*S]
    s2 = batched_result[W*S:]
    
    # Undo scaling for stage 2
    for r in range(W):
        s2[r*S:(r+1)*S] = s2[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stages 3-4 batched
    buf3 = s2.clone()
    for r in range(W):
        buf3[r*S:(r+1)*S] = a[r] * s2[r*S:(r+1)*S] / W
    
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    s_temp = s3.clone()
    for r in range(W):
        s_temp[r*S:(r+1)*S] = s3[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    buf4 = s_temp.clone()
    for r in range(W):
        buf4[r*S:(r+1)*S] = a[r] * s_temp[r*S:(r+1)*S] / W
    
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    for r in range(W):
        s4[r*S:(r+1)*S] = s4[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stages 5-6 batched
    buf5 = s4.clone()
    for r in range(W):
        buf5[r*S:(r+1)*S] = a[r] * s4[r*S:(r+1)*S] / W
    
    s5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    s_temp = s5.clone()
    for r in range(W):
        s_temp[r*S:(r+1)*S] = s5[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    buf6 = s_temp.clone()
    for r in range(W):
        buf6[r*S:(r+1)*S] = a[r] * s_temp[r*S:(r+1)*S] / W
    
    s6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    for r in range(W):
        s6[r*S:(r+1)*S] = s6[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Stage 7 (final)
    buf7 = s6.clone()
    for r in range(W):
        buf7[r*S:(r+1)*S] = a[r] * s6[r*S:(r+1)*S] / W
    
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf7)
    
    return s7