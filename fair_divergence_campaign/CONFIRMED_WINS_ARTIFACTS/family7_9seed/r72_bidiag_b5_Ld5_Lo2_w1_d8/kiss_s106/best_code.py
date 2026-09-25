
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C arrays
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.18 + 0.01 * (r % 5)
        sd = (r + 0) % 5
        for j in range(5):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific values
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.8 + 0.02 * rank
    wo = 0.18 + 0.01 * (rank % 5)
    
    # 6 iterations
    for iter in range(6):
        # Build buffer: start with base term, add offset where needed
        buf = wd * s
        for b in range(5):
            if b in ko and b >= 1:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Tridiagonal solve (skip on last iteration)
        if iter < 5:
            rec_parts = []
            rec_parts.append(acc[0:S] / A[0])
            for b in range(1, 5):
                prev_part = rec_parts[b-1]
                rec_parts.append((acc[b*S:(b+1)*S] - C[b] * prev_part) / A[b])
            s = torch.cat(rec_parts, dim=0)
        else:
            s = acc
    
    return s
