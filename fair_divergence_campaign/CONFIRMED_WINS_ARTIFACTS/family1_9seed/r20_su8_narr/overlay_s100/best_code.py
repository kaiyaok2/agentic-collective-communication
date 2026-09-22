def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute scaling factors to reduce redundant operations
    # After analyzing the pattern: scale by a[r]/W, all-reduce, unscale by a[r], repeat
    # This means: scale → AR → unscale → scale → AR is equivalent to fewer operations
    
    # Since unscale(a[r]) * scale(a[r]) = identity, we can simplify
    # The net effect after multiple stages is just repeated division by W with all-reduces
    
    # Combine all 7 stages into fewer collective calls by batching local ops
    # Stage 1
    for r in range(W):
        s[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    # Stages 2-7: Each does unscale, scale, all-reduce
    # Combine the unscale-scale pairs which simplify to division by W
    for stage in range(6):
        for r in range(W):
            chunk = s[r*S:(r+1)*S]
            # unscale then scale: (chunk / a[r]) * (a[r] / W) = chunk / W
            s[r*S:(r+1)*S] = chunk / W
        s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s