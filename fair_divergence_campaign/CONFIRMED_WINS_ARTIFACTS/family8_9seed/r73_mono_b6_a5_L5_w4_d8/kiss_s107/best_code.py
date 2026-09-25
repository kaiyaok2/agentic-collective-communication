
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Use all_gather + sum for first reduction
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    s = gathered.sum(dim=0)
    
    A = [0.0]*6
    for r in range(W):
        w_r = 0.45 + 0.02*(r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1*j) % 6] += w_r
    
    start = (rank + 1) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.45 + 0.02*(rank % 4)
    
    # Try all_gather + sum for iterations too
    for iteration in range(7):
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        
        # Use all_gather + sum instead of all_reduce
        gathered_buf = xm.all_gather(buf.unsqueeze(0), dim=0)
        acc = gathered_buf.sum(dim=0)
        
        if iteration < 6:
            rec = acc.clone()
            for b in range(6):
                jb = SIG[b]
                rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
            s = rec
        else:
            s = acc
    
    return s
