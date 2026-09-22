def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    B = 5
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        st = r % B
        ks = set((st + j) % B for j in range(2))
        for b in ks:
            c[b] += 1
    
    # Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine this rank's window
    start = rank % B
    keep = set((start + j) % B for j in range(2))
    
    # Strategy: Group the 7 dependent layers into parallel operations
    # Since each layer does the same window-based masking, we can
    # recognize that we're essentially doing multiple rounds of
    # the same selective reduction pattern
    
    # Layers 1-2: First group
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    # Layers 3-4: Second group
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    # Layers 5-6: Third group
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    
    # Layer 7: Final layer (no normalization)
    buf = torch.zeros_like(acc)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = acc[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s