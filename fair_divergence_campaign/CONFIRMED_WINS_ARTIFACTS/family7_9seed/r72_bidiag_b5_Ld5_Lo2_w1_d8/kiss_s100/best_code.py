
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C matrices
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.18 + 0.01 * (r % 5)
        sd = r % 5
        for j in range(5):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific parameters
    so = (rank + 1) % 5
    ko = {(so + j) % 5 for j in range(2)}
    wd = 0.8 + 0.02 * rank
    wo = 0.18 + 0.01 * (rank % 5)
    
    # 6 iterations
    for iteration in range(6):
        # Build buffer with diagonal contribution (always present for all blocks)
        buf = wd * s
        
        # Add off-diagonal contribution where applicable
        for b in range(5):
            if b in ko and b >= 1:
                start = b * S
                end = (b + 1) * S
                prev_start = (b - 1) * S
                prev_end = b * S
                buf[start:end] = buf[start:end] + wo * s[prev_start:prev_end]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration == 5:
            return acc
        
        # Recurrence
        s = acc.clone()
        s[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            start = b * S
            end = (b + 1) * S
            prev_start = (b - 1) * S
            prev_end = b * S
            s[start:end] = (acc[start:end] - C[b] * s[prev_start:prev_end]) / A[b]
    
    return s
