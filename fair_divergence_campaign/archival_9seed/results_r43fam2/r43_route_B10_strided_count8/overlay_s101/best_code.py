def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 2
    
    # Precompute per-block overlap count c[b] = #ranks whose window covers block b
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Precompute keep-set mask for THIS rank (used in all iterations)
    start = (rank + OFF) % B
    keep = set((start + STR * j) % B for j in range(L))
    
    # Initial all_reduce (dispatch 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reduced to 3 masked iterations (dispatches 2-4)
    for iteration in range(3):
        # Apply mask: zero out blocks not in keep-set
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale by per-block overlap count
        for b in range(B):
            acc[b * S:(b + 1) * S] = acc[b * S:(b + 1) * S] / c[b]
        
        s = acc
    
    # Final iteration (dispatch 5, no scaling needed as per reference)
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s