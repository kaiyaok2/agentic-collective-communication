
def r72_bidiag_b5_Ld4_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute coefficients A and C
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
    
    # Precompute rank-specific values
    sd = (rank + 1) % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 2) % 5
    ko = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    wd = 0.7 + 0.03 * (rank % 5)
    wo = 0.2 + 0.01 * (rank % 7)
    
    # Precompute weighted mask for kd blocks
    mask_d = torch.zeros_like(s)
    for b in kd:
        mask_d[b*S:(b+1)*S] = wd
    
    # Perform 6 iterations with solve
    for _ in range(6):
        # Use view for more efficient buffer construction
        s_blocks = s.view(5, S)
        mask_blocks = mask_d.view(5, S)
        buf_blocks = s_blocks * mask_blocks
        for b in ko:
            buf_blocks[b] += wo * s_blocks[b-1]
        buf = buf_blocks.view(-1)
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve using view
        acc_blocks = acc.view(5, S)
        acc_blocks[0] /= A[0]
        for b in range(1, 5):
            acc_blocks[b] = (acc_blocks[b] - C[b] * acc_blocks[b-1]) / A[b]
        s = acc_blocks.view(-1)
    
    # Final iteration without solve
    s_blocks = s.view(5, S)
    mask_blocks = mask_d.view(5, S)
    buf_blocks = s_blocks * mask_blocks
    for b in ko:
        buf_blocks[b] += wo * s_blocks[b-1]
    buf = buf_blocks.view(-1)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
