def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 8
    W = world_size
    L = 2
    OFF = 2
    
    # Precompute overlap counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Precompute this rank's keep set (reused across all iterations)
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Initial all_reduce to sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Launch 7 pipelined all_reduce operations with overlapping computation
    # We'll use a loop to avoid code duplication while maintaining async behavior
    
    for iteration in range(7):
        # Mask: keep only blocks in this rank's window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce the masked buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by its overlap count (only for iterations 0-5)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s