
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Precompute constants
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + j) % 6] += w
    
    start = (rank + 1) % 6
    keep = set((start + j) % 6 for j in range(5))
    my_weight = 0.45 + 0.02 * (rank % 4)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for round_idx in range(6):
        weighted_s = my_weight * s
        buf = torch.zeros_like(s)
        
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = weighted_s[jb*S:(jb+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if round_idx == 5:
            return acc
        
        # Create scale tensor for normalization
        scale = torch.zeros_like(s)
        for b in range(6):
            scale[b*S:(b+1)*S] = 1.0 / A[b]
        
        # Normalize all at once
        scaled_acc = acc * scale
        
        rec = torch.zeros_like(s)
        for b in range(6):
            jb = SIG[b]
            rec[jb*S:(jb+1)*S] = scaled_acc[b*S:(b+1)*S]
        
        s = rec
    
    return s
