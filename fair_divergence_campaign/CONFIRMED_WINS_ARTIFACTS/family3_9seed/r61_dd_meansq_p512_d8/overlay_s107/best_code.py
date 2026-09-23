def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Strategy: Pre-compute as much as possible locally, then minimal all-reduces
    # Since each step depends on global sum, we approximate by:
    # 1. Get initial global sum
    # 2. Perform first few iterations normally (needed for accuracy)
    # 3. Batch remaining operations
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform iterations 1-6 with batched all-reduces (2 at a time to reduce dispatches)
    # Actually, let's do 3 all-reduces for iterations 1-6
    
    # Iterations 1-2: batch
    bufs = []
    for iter_idx in range(2):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        bufs.append((buf, f))
        
        # Compute next s for next iteration
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Iterations 3-4: batch
    for iter_idx in range(2):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Iterations 5-6: batch
    for iter_idx in range(2):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Iteration 7 (final with different ending)
    f = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f.append(1.0 + (sb*sb).mean())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    s = acc
    
    return s