
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.8 + 0.02*r
        wo = 0.18 + 0.01*(r % 5)
        sd = (r + 0) % 5
        for j in range(5):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    wd = 0.8 + 0.02*rank
    wo = 0.18 + 0.01*(rank % 5)
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    
    # kd contains all 5 blocks, so we can just multiply s by wd
    buf = wd * s
    
    # Add off-diagonal contributions
    for b in range(5):
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
