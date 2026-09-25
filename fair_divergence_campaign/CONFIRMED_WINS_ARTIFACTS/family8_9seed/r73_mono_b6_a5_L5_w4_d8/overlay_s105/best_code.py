def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Compute A[b] - total weight for each block across all ranks
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Current rank's weight and block window
    w = 0.45 + 0.02 * (rank % 4)
    start = (rank + 1) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    
    # Dispatch 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Dispatches 2-7: Permute, weight, all-reduce, unpermute+normalize
    for iteration in range(6):
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        rec = acc.clone()
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = rec
    
    # Dispatch 8: Final permute, weight, all-reduce (no normalization)
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            jb = SIG[b]
            buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s