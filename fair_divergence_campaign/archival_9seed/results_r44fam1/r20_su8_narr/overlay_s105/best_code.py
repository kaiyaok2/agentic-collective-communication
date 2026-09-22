def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused kernel: clone + scale + divide by W (for all shards)
    # Instead of: buf = s.clone() + loop with buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    # We create a weight tensor and apply it in one operation
    weights = torch.zeros(W * S, dtype=dtype, device=x.device)
    for r in range(W):
        weights[r*S:(r+1)*S] = a[r] / W
    buf = s * weights
    
    # Stages 2-8: each stage does all-reduce, unscale, then scale for next iteration
    for stage in range(7):
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if stage < 6:  # Not the last stage
            # Fused kernel: unscale (divide by a[r]) + scale (multiply by a[r]) + divide by W
            # Unscale: s[r*S:(r+1)*S] / a[r]
            # Then scale and divide: result * a[r] / W
            # Net effect: s[r*S:(r+1)*S] / W for each shard
            weights_unscale = torch.zeros(W * S, dtype=dtype, device=x.device)
            for r in range(W):
                weights_unscale[r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)
            s = s * weights_unscale
            
            # Fused kernel: scale + divide by W
            weights_scale = torch.zeros(W * S, dtype=dtype, device=x.device)
            for r in range(W):
                weights_scale[r*S:(r+1)*S] = a[r] / W
            buf = s * weights_scale
    
    # Final all-reduce (8th)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s