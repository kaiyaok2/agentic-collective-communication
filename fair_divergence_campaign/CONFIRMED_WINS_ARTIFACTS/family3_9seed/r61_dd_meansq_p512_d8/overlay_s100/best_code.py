def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get sum across ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # We'll batch iterations to reduce all-reduce count
    # Reference does 7 iterations after initial all-reduce (each with scale-reduce-unscale pattern)
    # Strategy: batch 2-3 iterations together
    
    # Batch 1: iterations 1-3 (3 iterations)
    batch_size = 3
    batch_buf = torch.zeros((batch_size, B * S), dtype=dtype, device=x.device)
    
    for iter_idx in range(batch_size):
        # Compute scale factors for current state
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        
        # Apply scaling and store in batch buffer
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        batch_buf[iter_idx] = buf
        
        # Speculatively compute next state (approximation for batching)
        # We assume the all-reduce would give us world_size * buf
        # Then unscale: world_size * buf / (world_size * f[b]) = buf / f[b]
        acc = buf.clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / f[b]
        s = acc
    
    # Issue batched all-reduce
    batch_buf_flat = batch_buf.reshape(-1)
    batch_reduced = xm.all_reduce(xm.REDUCE_SUM, batch_buf_flat)
    batch_reduced = batch_reduced.reshape(batch_size, B * S)
    
    # Now recompute correct states from batched results
    # Start from original summed x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iter_idx in range(batch_size):
        # Compute scale factors
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        
        # Get the reduced buffer for this iteration
        acc = batch_reduced[iter_idx]
        
        # Unscale
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Batch 2: iterations 4-6 (3 iterations)
    batch_buf = torch.zeros((batch_size, B * S), dtype=dtype, device=x.device)
    
    for iter_idx in range(batch_size):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        batch_buf[iter_idx] = buf
        
        acc = buf.clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / f[b]
        s = acc
    
    # Issue second batched all-reduce
    batch_buf_flat = batch_buf.reshape(-1)
    batch_reduced = xm.all_reduce(xm.REDUCE_SUM, batch_buf_flat)
    batch_reduced = batch_reduced.reshape(batch_size, B * S)
    
    # Recompute from where we left off
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    for iter_idx in range(3):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        acc = s.clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        acc = xm.all_reduce(xm.REDUCE_SUM, acc)
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    for iter_idx in range(batch_size):
        f = []
        for b in range(B):
            sb = s[b*S:(b+1)*S]
            f.append(1.0 + (sb*sb).mean())
        acc = batch_reduced[iter_idx]
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Iteration 7 (final)
    f = []
    for b in range(B):
        sb = s[b*S:(b+1)*S]
        f.append(1.0 + (sb*sb).mean())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    
    return acc