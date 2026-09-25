
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    A = [0.0] * 6
    for r in range(W):
        w = 0.45 + 0.02 * (r % 4)
        start = (r + 1) % 6
        for j in range(5):
            A[(start + j) % 6] += w
    
    for iteration in range(6):
        start = (rank + 1) % 6
        keep = set((start + j) % 6 for j in range(5))
        w = 0.45 + 0.02 * (rank % 4)
        
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                jb = SIG[b]
                buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 5:
            s = torch.zeros_like(acc)
            for b in range(6):
                jb = SIG[b]
                s[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]
        else:
            s = acc
    
    return s
