def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # per-block overlap count
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Pre-compute keep blocks
    start = (rank + OFF) % B
    keep_blocks = [(start + j) % B for j in range(L)]
    non_keep_blocks = [b for b in range(B) if b not in keep_blocks]
    
    # Allocate buffer once
    buf = torch.zeros_like(s)
    
    # Try 2 iterations
    for iteration in range(2):
        for b in non_keep_blocks:
            buf[b*S:(b+1)*S] = 0
        for b in keep_blocks:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 1:  # Normalize on first iteration only
            for b in range(B):
                if c[b] > 0:
                    acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s