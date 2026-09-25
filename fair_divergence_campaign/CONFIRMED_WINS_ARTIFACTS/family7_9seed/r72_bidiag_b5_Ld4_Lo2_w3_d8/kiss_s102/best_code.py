
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute A and C once
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.9 + 0.04 * (r % 3)
        wo = 0.25 + 0.01 * (r % 3)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Pre-compute rank-specific constants
    sd = rank % 5
    so = (rank + 1) % 5
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    
    # Determine which blocks to process
    kd_blocks = [(sd + j) % 5 for j in range(4)]
    ko_blocks = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    
    # Work with 2D view throughout
    s = s.view(5, S)
    
    # Main iteration loop (6 iterations with solve)
    for _ in range(6):
        # Build buffer
        buf = torch.zeros_like(s)
        for b in kd_blocks:
            buf[b] = wd * s[b]
        for b in ko_blocks:
            buf[b] += wo * s[b-1]
        
        # All-reduce (flatten for communication)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        acc = acc.view(5, S)
        
        # Solve tridiagonal system
        s = torch.zeros_like(acc)
        s[0] = acc[0] / A[0]
        for b in range(1, 5):
            s[b] = (acc[b] - C[b] * s[b-1]) / A[b]
    
    # Final iteration (no solve)
    buf = torch.zeros_like(s)
    for b in kd_blocks:
        buf[b] = wd * s[b]
    for b in ko_blocks:
        buf[b] += wo * s[b-1]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return s
