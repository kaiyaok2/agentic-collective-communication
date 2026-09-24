
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Pre-compute accumulated weights
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Current rank weight
    w = 0.45 + 0.02 * (rank % 4)
    
    # Keep set for this rank (same for all rounds)
    start = (rank + 1) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 rounds
    for round_idx in range(6):
        # Create buffer with selected/weighted buckets
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize (except last round)
        if round_idx < 5:
            rec = torch.zeros_like(acc)
            for b in range(6):
                jb = SIG[b]
                rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
            s = rec
        else:
            s = acc
    
    return s
