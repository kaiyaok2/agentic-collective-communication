def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # per-block overlap count c[b] = #ranks whose window covers block b
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Do 1 iteration
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    
    # Final iteration without division
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s