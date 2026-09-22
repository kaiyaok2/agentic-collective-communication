def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scale factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # We'll batch stages 2-4 and 5-8 together
    # Stages 2-4: 3 stages (apply scale, reduce, undo scale)
    # Stages 5-8: 4 stages (apply scale, reduce, undo scale, apply scale again)
    
    # Batch stages 2-4 (3 iterations of the pattern)
    # Each iteration: scale -> reduce -> unscale
    batch_1_stages = 3
    stacked_1 = torch.zeros((batch_1_stages, W * S), dtype=dtype, device=x.device)
    
    current_s = s.clone()
    for stage_idx in range(batch_1_stages):
        buf = current_s.clone()
        for r in range(W):
            buf[r*S:(r+1)*S] = a[r] * current_s[r*S:(r+1)*S] / W
        stacked_1[stage_idx] = buf
    
    # All-reduce the stacked buffer
    reduced_1 = xm.all_reduce(xm.REDUCE_SUM, stacked_1)
    
    # Unpack and process each stage
    for stage_idx in range(batch_1_stages):
        temp_s = reduced_1[stage_idx]
        for r in range(W):
            temp_s[r*S:(r+1)*S] = temp_s[r*S:(r+1)*S] / max(a[r], 1e-9)
        if stage_idx < batch_1_stages - 1:
            # Prepare for next stage in batch
            current_s = temp_s
    
    s = temp_s
    
    # Batch stages 5-8 (4 iterations of the pattern)
    batch_2_stages = 4
    stacked_2 = torch.zeros((batch_2_stages, W * S), dtype=dtype, device=x.device)
    
    current_s = s.clone()
    for stage_idx in range(batch_2_stages):
        buf = current_s.clone()
        for r in range(W):
            buf[r*S:(r+1)*S] = a[r] * current_s[r*S:(r+1)*S] / W
        stacked_2[stage_idx] = buf
    
    # All-reduce the stacked buffer
    reduced_2 = xm.all_reduce(xm.REDUCE_SUM, stacked_2)
    
    # Unpack and process each stage
    for stage_idx in range(batch_2_stages):
        temp_s = reduced_2[stage_idx]
        if stage_idx < batch_2_stages - 1:
            # Unscale for stages 5-7
            for r in range(W):
                temp_s[r*S:(r+1)*S] = temp_s[r*S:(r+1)*S] / max(a[r], 1e-9)
            current_s = temp_s
        else:
            # Stage 8: don't unscale, this is the final result
            s = temp_s
    
    return s