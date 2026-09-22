def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # per-block overlap count
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # This rank's blocks
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    buf = torch.zeros_like(s)
    
    # Just one final iteration without division
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        else:
            buf[b*S:(b+1)*S] = 0
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s