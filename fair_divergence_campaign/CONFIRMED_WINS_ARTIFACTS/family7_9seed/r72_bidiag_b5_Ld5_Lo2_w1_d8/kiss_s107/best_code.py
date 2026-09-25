
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C matrices
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.8 + 0.02*r
        wo = 0.18 + 0.01*(r % 5)
        sd = r % 5
        for j in range(5):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-dependent values
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.8 + 0.02*rank
    wo = 0.18 + 0.01*(rank % 5)
    
    # First 5 iterations with tridiagonal solve
    for _ in range(5):
        buf = wd * s  # All buckets get diagonal weight
        for b in range(5):
            if b in ko and b >= 1:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal in-place
        acc[0:S] = acc[0:S] / A[0]
        for b in range(1, 5):
            acc[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * acc[(b-1)*S:b*S]) / A[b]
        s = acc
    
    # Last iteration without tridiagonal solve
    buf = wd * s
    for b in range(5):
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
