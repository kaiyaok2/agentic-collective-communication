
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.9 + 0.04*(r % 3)
        wo = 0.25 + 0.01*(r % 3)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Compute rank-specific parameters
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd_val = 0.9 + 0.04*(rank % 3)
    wo_val = 0.25 + 0.01*(rank % 3)
    
    # Precompute block operations
    blocks_both = [b for b in range(5) if b in kd and b in ko and b >= 1]
    blocks_diag = [b for b in range(5) if b in kd and not (b in ko and b >= 1)]
    blocks_offdiag = [b for b in range(5) if b in ko and b >= 1 and b not in kd]
    
    # Do 5 iterations with the tridiagonal solve
    for iteration in range(5):
        buf = torch.zeros_like(s)
        s_view = s.view(5, S)
        buf_view = buf.view(5, S)
        for b in blocks_both:
            buf_view[b] = wd_val * s_view[b] + wo_val * s_view[b-1]
        for b in blocks_diag:
            buf_view[b] = wd_val * s_view[b]
        for b in blocks_offdiag:
            buf_view[b] = wo_val * s_view[b-1]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Use view for tridiagonal solve
        s = torch.zeros_like(acc)
        s_view = s.view(5, S)
        acc_view = acc.view(5, S)
        s_view[0] = acc_view[0] / A[0]
        s_view[1] = (acc_view[1] - C[1] * s_view[0]) / A[1]
        s_view[2] = (acc_view[2] - C[2] * s_view[1]) / A[2]
        s_view[3] = (acc_view[3] - C[3] * s_view[2]) / A[3]
        s_view[4] = (acc_view[4] - C[4] * s_view[3]) / A[4]
    
    # Final iteration without the tridiagonal solve
    buf = torch.zeros_like(s)
    s_view = s.view(5, S)
    buf_view = buf.view(5, S)
    for b in blocks_both:
        buf_view[b] = wd_val * s_view[b] + wo_val * s_view[b-1]
    for b in blocks_diag:
        buf_view[b] = wd_val * s_view[b]
    for b in blocks_offdiag:
        buf_view[b] = wo_val * s_view[b-1]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
