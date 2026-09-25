
def r72_bidiag_b5_Ld5_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute coefficients A and C
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
    
    # Precompute rank-specific data
    so = (rank + 1) % 5
    ko = [(so + j) % 5 for j in range(2)]
    wd = 0.8 + 0.02 * rank
    wo = 0.18 + 0.01 * (rank % 5)
    
    for iteration in range(6):
        # Create buffer: all blocks get wd * s
        buf = wd * s
        
        # Add off-diagonal contributions
        for b in ko:
            if b >= 1:
                buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal system (all iterations except the last)
        if iteration < 5:
            # Build result slice by slice to avoid clone
            slices = []
            slice_0 = acc[0:S] / A[0]
            slices.append(slice_0)
            for b in range(1, 5):
                slice_b = (acc[b*S:(b+1)*S] - C[b] * slices[b-1]) / A[b]
                slices.append(slice_b)
            s = torch.cat(slices, dim=0)
    
    return acc
