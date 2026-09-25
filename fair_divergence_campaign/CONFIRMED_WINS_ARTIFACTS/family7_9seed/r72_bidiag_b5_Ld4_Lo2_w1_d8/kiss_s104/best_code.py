def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C once
    A = [0.0]*5
    C = [0.0]*5
    for r in range(W):
        wd = 0.8 + 0.02*r
        wo = 0.15 + 0.01*(r % 5)
        sd = r % 5
        for j in range(4):
            A[(sd + j) % 5] += wd
        so = (r + 1) % 5
        for j in range(2):
            b = (so + j) % 5
            if b >= 1:
                C[b] += wo
    
    # Precompute rank-specific values
    sd = rank % 5
    kd = set((sd + j) % 5 for j in range(4))
    so = (rank + 1) % 5
    ko = set((so + j) % 5 for j in range(2)) & {1, 2, 3, 4}
    wd = 0.8 + 0.02*rank
    wo = 0.15 + 0.01*(rank % 5)
    
    blocks_both = sorted(kd & ko)
    blocks_kd_only = sorted(kd - ko)
    blocks_ko_only = sorted(ko - kd)
    
    # Helper function to build buffer
    def build_buf(s):
        buf = torch.zeros_like(s)
        for b in blocks_both:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        for b in blocks_kd_only:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        for b in blocks_ko_only:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        return buf
    
    # Helper function for tridiagonal solve
    def solve(acc):
        r0 = acc[0:S] / A[0]
        r1 = (acc[S:2*S] - C[1] * r0) / A[1]
        r2 = (acc[2*S:3*S] - C[2] * r1) / A[2]
        r3 = (acc[3*S:4*S] - C[3] * r2) / A[3]
        r4 = (acc[4*S:5*S] - C[4] * r3) / A[4]
        return torch.cat([r0, r1, r2, r3, r4])
    
    # First 5 iterations with solve
    for _ in range(5):
        acc = xm.all_reduce(xm.REDUCE_SUM, build_buf(s))
        s = solve(acc)
    
    # Last iteration without solve
    s = xm.all_reduce(xm.REDUCE_SUM, build_buf(s))
    
    return s