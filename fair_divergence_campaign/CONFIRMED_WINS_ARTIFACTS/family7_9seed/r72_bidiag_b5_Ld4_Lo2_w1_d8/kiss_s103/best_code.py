
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C coefficients once
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
    
    # Precompute rank-specific parameters
    sd = rank % 5
    kd = {(sd + j) % 5 for j in range(4)}
    so = (rank + 1) % 5
    ko = {(so + j) % 5 for j in range(2)}
    wd = 0.8 + 0.02 * rank
    wo = 0.15 + 0.01 * (rank % 5)
    
    # Perform 7 iterations
    for iteration in range(7):
        # Build buffer more efficiently by combining conditions
        buf = torch.zeros_like(s)
        for b in range(5):
            start = b * S
            end = (b + 1) * S
            in_kd = b in kd
            in_ko = b in ko and b >= 1
            
            if in_kd and in_ko:
                # Both contributions
                buf[start:end] = wd * s[start:end] + wo * s[(b-1)*S:b*S]
            elif in_kd:
                # Only diagonal
                buf[start:end] = wd * s[start:end]
            elif in_ko:
                # Only off-diagonal
                buf[start:end] = wo * s[(b-1)*S:b*S]
            # else: leave as zero
        
        # All_reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # For the last iteration, just return the result
        if iteration == 6:
            return acc
        
        # Triangular solve (forward substitution)
        b0 = acc[0:S] / A[0]
        b1 = (acc[S:2*S] - C[1] * b0) / A[1]
        b2 = (acc[2*S:3*S] - C[2] * b1) / A[2]
        b3 = (acc[3*S:4*S] - C[3] * b2) / A[3]
        b4 = (acc[4*S:5*S] - C[4] * b3) / A[4]
        
        s = torch.cat([b0, b1, b2, b3, b4], dim=0)
    
    return s
