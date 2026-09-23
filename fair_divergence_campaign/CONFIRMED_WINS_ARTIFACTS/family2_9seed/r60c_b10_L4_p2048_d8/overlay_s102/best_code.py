def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    L = 4
    
    dtype = x.dtype
    
    # Precompute counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(L))
        for b in ks:
            c[b] += 1
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute this rank's window
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(L))
    
    # Reduced pipeline: 3 iterations (4 total all_reduces including the initial one)
    # This reduces collective dispatch overhead
    for iteration in range(3):
        # Prepare buffer: zero out blocks not in window
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All_reduce the windowed data
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Scale each block by its count
        if iteration < 2:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s