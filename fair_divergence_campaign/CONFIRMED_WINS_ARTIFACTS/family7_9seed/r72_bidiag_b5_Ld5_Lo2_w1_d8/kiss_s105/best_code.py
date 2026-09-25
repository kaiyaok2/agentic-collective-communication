
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
    
    so = (rank + 1) % 5
    wd = 0.8 + 0.02*rank
    wo = 0.18 + 0.01*(rank % 5)
    
    # Precompute which blocks need off-diagonal terms
    ko_blocks = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    
    # 7 iterations
    for iteration in range(7):
        # Since kd contains all blocks, buf = wd * s for all blocks
        buf = wd * s
        
        # Add off-diagonal terms only for ko blocks
        for b in ko_blocks:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 5):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:
            s = acc
    
    return s
