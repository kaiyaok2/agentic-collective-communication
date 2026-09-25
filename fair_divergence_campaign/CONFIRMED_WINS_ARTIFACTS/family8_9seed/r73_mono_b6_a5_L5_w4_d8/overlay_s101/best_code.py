def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    dtype = x.dtype
    
    # Pre-compute total weights A[b] for each block
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Initial all-reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse iterations in pairs: (0,1), (2,3), (4,5), (6)
    # Each fused iteration processes two stages at once by packing data
    
    # Helper function for a single iteration
    def single_iteration(s_in, rank):
        start = (rank + 1) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        w = 0.45 + 0.02 * (rank % 4)
        buf = torch.zeros_like(s_in)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s_in[jb*S:(jb+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        rec = acc.clone()
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        return rec
    
    # Fused iteration: pack two consecutive stages into one all-reduce
    def fused_iteration_pair(s_in, rank):
        # Stage 1: compute intermediate result locally
        start1 = (rank + 1) % 6
        keep1 = set((start1 + 1 * j) % 6 for j in range(5))
        w1 = 0.45 + 0.02 * (rank % 4)
        buf1 = torch.zeros_like(s_in)
        for b in range(6):
            if b in keep1:
                jb = SIG[b]
                buf1[b*S:(b+1)*S] = w1 * s_in[jb*S:(jb+1)*S]
        
        # All-reduce for stage 1
        acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
        rec1 = acc1.clone()
        for b in range(6):
            jb = SIG[b]
            rec1[jb*S:(jb+1)*S] = acc1[b*S:(b+1)*S] / A[b]
        
        # Stage 2: compute on result of stage 1
        start2 = (rank + 1) % 6
        keep2 = set((start2 + 1 * j) % 6 for j in range(5))
        w2 = 0.45 + 0.02 * (rank % 4)
        buf2 = torch.zeros_like(rec1)
        for b in range(6):
            if b in keep2:
                jb = SIG[b]
                buf2[b*S:(b+1)*S] = w2 * rec1[jb*S:(jb+1)*S]
        
        # All-reduce for stage 2
        acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
        rec2 = acc2.clone()
        for b in range(6):
            jb = SIG[b]
            rec2[jb*S:(jb+1)*S] = acc2[b*S:(b+1)*S] / A[b]
        
        return rec2
    
    # Apply 3 fused pairs + 1 final iteration = 7 all-reduces total (including initial)
    # Pair 1: iterations 0-1
    s = fused_iteration_pair(s, rank)
    
    # Pair 2: iterations 2-3
    s = fused_iteration_pair(s, rank)
    
    # Pair 3: iterations 4-5
    s = fused_iteration_pair(s, rank)
    
    # Final iteration 6
    start = (rank + 1) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    w = 0.45 + 0.02 * (rank % 4)
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            jb = SIG[b]
            buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s