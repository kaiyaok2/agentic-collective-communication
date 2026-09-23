def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Compute scaling factors a[r] = 1.0 + 0.4 * (r % 5) for each rank
    a = torch.tensor([1.0 + 0.4 * (r % 5) for r in range(W)], dtype=dtype, device=x.device)
    
    # Stage 1: Initial all-reduce to sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute scaling vectors for efficiency
    a_expanded = a.repeat_interleave(S)
    
    # Stages 2-8: Apply per-shard scaling, reduce, then undo scaling
    for stage in range(2, 9):
        # Apply per-shard scaling a[r] and normalize by W in one vectorized op
        buf = (s * a_expanded) / W
        
        # All-reduce the scaled buffer
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Undo the per-shard scaling (except for the last stage) in vectorized form
        if stage < 8:
            s = s / torch.clamp(a_expanded, min=1e-9)
    
    return s