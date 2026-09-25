def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Precompute A coefficients
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Stage 1: Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Key optimization: Fuse stages 2-8 into fewer collective operations
    # Instead of 7 separate all_reduces, we'll batch them together
    
    # Compute all 7 stages' local operations first, then do fewer collectives
    # We'll use a buffer that holds multiple stages' data
    
    w = 0.45 + 0.02 * (rank % 4)
    start = (rank + 1) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    
    # Process stages in groups to reduce collective count
    # Group stages: 2-4, 5-7, 8
    stage_groups = [[0, 1, 2], [3, 4, 5], [6]]
    
    for group_idx, stages in enumerate(stage_groups):
        if group_idx == 0:
            # First group: stages 2-4 (indices 0-2)
            num_stages = len(stages)
            # Allocate buffer for multiple stages
            multi_buf = torch.zeros(num_stages, 6 * S, dtype=s.dtype, device=s.device)
            
            for i, stage in enumerate(stages):
                buf = torch.zeros_like(s)
                for b in range(6):
                    if b in keep:
                        jb = SIG[b]
                        buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
                multi_buf[i] = buf
            
            # Single all_reduce for all 3 stages
            multi_acc = xm.all_reduce(xm.REDUCE_SUM, multi_buf)
            
            # Process results sequentially
            for i in range(num_stages):
                acc = multi_acc[i]
                rec = acc.clone()
                for b in range(6):
                    jb = SIG[b]
                    rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
                s = rec
        
        elif group_idx == 1:
            # Second group: stages 5-7 (indices 3-5)
            num_stages = len(stages)
            multi_buf = torch.zeros(num_stages, 6 * S, dtype=s.dtype, device=s.device)
            
            for i, stage in enumerate(stages):
                buf = torch.zeros_like(s)
                for b in range(6):
                    if b in keep:
                        jb = SIG[b]
                        buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
                multi_buf[i] = buf
            
            # Single all_reduce for all 3 stages
            multi_acc = xm.all_reduce(xm.REDUCE_SUM, multi_buf)
            
            for i in range(num_stages):
                acc = multi_acc[i]
                rec = acc.clone()
                for b in range(6):
                    jb = SIG[b]
                    rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
                s = rec
        
        else:
            # Final stage: stage 8 (index 6)
            buf = torch.zeros_like(s)
            for b in range(6):
                if b in keep:
                    jb = SIG[b]
                    buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
            
            s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s