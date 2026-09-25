
def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C arrays
    A = [0.0] * 6
    C = [0.0] * 6
    for r in range(W):
        wd = 0.7 + 0.03 * (r % 5)
        wo = 0.2 + 0.01 * (r % 7)
        sd = (r + 2) % 6
        for j in range(5):
            A[(sd + j) % 6] += wd
        so = (r + 3) % 6
        for j in range(2):
            b = (so + j) % 6
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific parameters
    sd = (rank + 2) % 6
    kd = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko = set((so + j) % 6 for j in range(2))
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Main iteration loop - 6 iterations with solving
    for _ in range(6):
        # Create buffer and compute weighted contributions
        buf = torch.zeros_like(s)
        for b in range(6):
            start = b * S
            end = (b + 1) * S
            # Combine both conditions into fewer operations
            if b in kd and b in ko and b >= 1:
                prev_start = (b - 1) * S
                prev_end = b * S
                buf[start:end] = wd * s[start:end] + wo * s[prev_start:prev_end]
            elif b in kd:
                buf[start:end] = wd * s[start:end]
            elif b in ko and b >= 1:
                prev_start = (b - 1) * S
                prev_end = b * S
                buf[start:end] = wo * s[prev_start:prev_end]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal system directly into new tensor (no clone)
        s = torch.zeros_like(acc)
        s[0:S] = acc[0:S] / A[0]
        for b in range(1, 6):
            start = b * S
            end = (b + 1) * S
            prev_start = (b - 1) * S
            prev_end = b * S
            s[start:end] = (acc[start:end] - C[b] * s[prev_start:prev_end]) / A[b]
    
    # Final iteration without solving
    buf = torch.zeros_like(s)
    for b in range(6):
        start = b * S
        end = (b + 1) * S
        if b in kd and b in ko and b >= 1:
            prev_start = (b - 1) * S
            prev_end = b * S
            buf[start:end] = wd * s[start:end] + wo * s[prev_start:prev_end]
        elif b in kd:
            buf[start:end] = wd * s[start:end]
        elif b in ko and b >= 1:
            prev_start = (b - 1) * S
            prev_end = b * S
            buf[start:end] = wo * s[prev_start:prev_end]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
