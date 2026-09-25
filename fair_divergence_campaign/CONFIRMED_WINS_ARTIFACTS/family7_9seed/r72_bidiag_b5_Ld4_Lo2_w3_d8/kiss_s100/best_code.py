
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Precompute global weights A and C
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
    
    # Precompute rank-specific parameters
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Main iteration loop - 7 iterations
    for iteration in range(7):
        # Apply weighted bucket operation
        buf = torch.zeros_like(s)
        for b in range(5):
            start = b * S
            end = (b + 1) * S
            if b in kd:
                buf[start:end] = wd * s[start:end]
            if b in ko and b >= 1:
                prev_start = (b - 1) * S
                prev_end = b * S
                buf[start:end] = buf[start:end] + wo * s[prev_start:prev_end]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Tridiagonal solve (except last iteration)
        if iteration < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 5):
                start = b * S
                end = (b + 1) * S
                prev_start = (b - 1) * S
                prev_end = b * S
                rec[start:end] = (acc[start:end] - C[b] * rec[prev_start:prev_end]) / A[b]
            s = rec
        else:
            s = acc
    
    return s
