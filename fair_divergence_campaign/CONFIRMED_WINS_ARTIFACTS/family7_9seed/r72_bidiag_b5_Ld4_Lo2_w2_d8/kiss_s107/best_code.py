
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute global coefficients A and C
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.7 + 0.03 * (r % 5)
        wo = 0.2 + 0.01 * (r % 7)
        sd = (r + 1) % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 2) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Pre-compute rank-specific parameters
    sd = (rank + 1) % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 2) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Pre-compute block categories
    both = kd & ko
    only_kd = kd - ko
    only_ko = ko - kd
    
    # Perform 6 iterations
    for iteration in range(6):
        # Work with blocks using view
        s_view = s.view(5, S)
        buf = torch.zeros_like(s)
        buf_view = buf.view(5, S)
        
        # Blocks only in kd
        for b in only_kd:
            buf_view[b] = wd * s_view[b]
        
        # Blocks only in ko
        for b in only_ko:
            if b >= 1:
                buf_view[b] = wo * s_view[b-1]
        
        # Blocks in both
        for b in both:
            if b >= 1:
                buf_view[b] = wd * s_view[b] + wo * s_view[b-1]
            else:
                buf_view[b] = wd * s_view[b]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 5:
            acc_view = acc.view(5, S)
            acc_view[0] = acc_view[0] / A[0]
            for b in range(1, 5):
                acc_view[b] = (acc_view[b] - C[b] * acc_view[b-1]) / A[b]
        
        s = acc
    
    return s
