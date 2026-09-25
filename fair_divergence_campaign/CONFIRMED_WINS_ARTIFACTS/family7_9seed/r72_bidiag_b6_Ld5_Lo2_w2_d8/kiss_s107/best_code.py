
def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute global coefficients A and C once
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
    
    # Compute rank-specific constants once
    sd = (rank + 2) % 6
    kd_list = [(sd + j) % 6 for j in range(5)]
    so = (rank + 3) % 6
    ko_list = [(so + j) % 6 for j in range(2) if (so + j) % 6 >= 1]
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # 7 iterations
    for iteration in range(7):
        # Construct buffer
        buf = torch.zeros_like(s)
        for b in kd_list:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        for b in ko_list:
            buf[b*S:(b+1)*S] += wo * s[(b-1)*S:b*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve (except last iteration)
        if iteration < 6:
            rec = acc.clone()
            rec[0:S] = acc[0:S] / A[0]
            for b in range(1, 6):
                rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]
            s = rec
        else:
            s = acc
    
    return s
