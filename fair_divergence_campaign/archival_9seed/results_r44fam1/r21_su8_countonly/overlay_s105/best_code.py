def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Pipelined two-stage batching: Group 8 all-reduces into 2 batches of 4
    # Batch 1: all-reduces 1-4
    # Batch 2: all-reduces 5-8
    
    # Batch 1, Step 1: all-reduce 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Prepare buffer for all-reduce 2 while s is ready
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 1, Step 2: all-reduce 2
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse scaling and prepare buffer for all-reduce 3
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 1, Step 3: all-reduce 3
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse scaling and prepare buffer for all-reduce 4
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 1, Step 4: all-reduce 4
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # End of Batch 1 - prepare for Batch 2
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 2, Step 1: all-reduce 5
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse scaling and prepare buffer for all-reduce 6
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 2, Step 2: all-reduce 6
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse scaling and prepare buffer for all-reduce 7
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 2, Step 3: all-reduce 7
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Inverse scaling and prepare buffer for all-reduce 8
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    
    # Batch 2, Step 4: all-reduce 8 (final)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s