
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    
    # Calculate which buckets this rank keeps
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    # Calculate counts for normalization
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(5))
        for b in ks:
            c[b] += 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First iteration: mask, reduce, normalize only kept buckets
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Normalize and prepare for final reduce in one pass
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            if c[b] > 0:
                buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
            else:
                buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
