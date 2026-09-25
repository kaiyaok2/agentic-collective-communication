
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute global coefficients
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(W):
        wd = 0.8 + 0.02 * r
        wo = 0.15 + 0.01 * (r % 5)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute reciprocals and rank-specific values
    inv_A = [1.0 / a for a in A]
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2))
    wd = 0.8 + 0.02 * rank
    wo = 0.15 + 0.01 * (rank % 5)
    
    # Main iteration loop
    for iteration in range(6):
        # Build weighted buffer with cached slices
        s_blocks = [s[b*S:(b+1)*S] for b in range(5)]
        buf = torch.zeros_like(s)
        
        for b in range(5):
            in_kd = b in kd
            in_ko = b in ko and b >= 1
            if in_kd and in_ko:
                buf[b*S:(b+1)*S] = wd * s_blocks[b] + wo * s_blocks[b-1]
            elif in_kd:
                buf[b*S:(b+1)*S] = wd * s_blocks[b]
            elif in_ko:
                buf[b*S:(b+1)*S] = wo * s_blocks[b-1]
        
        # All-reduce the buffer
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve or just use acc (last iteration skips solve)
        if iteration < 5:
            # Precompute acc blocks
            acc_blocks = [acc[b*S:(b+1)*S] for b in range(5)]
            
            # Forward substitution with torch.cat
            rec_blocks = [acc_blocks[0] * inv_A[0]]
            for b in range(1, 5):
                prev_block = rec_blocks[b-1]
                curr_block = (acc_blocks[b] - C[b] * prev_block) * inv_A[b]
                rec_blocks.append(curr_block)
            
            s = torch.cat(rec_blocks, dim=0)
        else:
            s = acc
    
    return s
