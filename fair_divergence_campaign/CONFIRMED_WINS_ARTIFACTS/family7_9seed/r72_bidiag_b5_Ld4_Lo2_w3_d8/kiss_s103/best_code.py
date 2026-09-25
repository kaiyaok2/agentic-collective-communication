
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Precompute A and C arrays
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
    
    # Rank-specific setup - optimize block lists
    sd = rank % 5
    so = (rank + 1) % 5
    wd = 0.9 + 0.04*(rank % 3)
    wo = 0.25 + 0.01*(rank % 3)
    
    # Precompute which blocks get which contributions
    kd_list = [(sd + j) % 5 for j in range(4)]
    ko_list = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    
    # Find blocks that need both (intersection)
    kd_set = set(kd_list)
    ko_set = set(ko_list)
    both = list(kd_set & ko_set)
    only_d = list(kd_set - ko_set)
    only_o = list(ko_set - kd_set)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Main loop
    for iteration in range(7):
        buf = torch.zeros_like(s)
        
        # Blocks with only diagonal contribution
        for b in only_d:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
        
        # Blocks with only off-diagonal contribution  
        for b in only_o:
            buf[b*S:(b+1)*S] = wo * s[(b-1)*S:b*S]
        
        # Blocks with both contributions
        for b in both:
            buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            acc[0:S] = acc[0:S] / A[0]
            acc[S:2*S] = (acc[S:2*S] - C[1] * acc[0:S]) / A[1]
            acc[2*S:3*S] = (acc[2*S:3*S] - C[2] * acc[S:2*S]) / A[2]
            acc[3*S:4*S] = (acc[3*S:4*S] - C[3] * acc[2*S:3*S]) / A[3]
            acc[4*S:5*S] = (acc[4*S:5*S] - C[4] * acc[3*S:4*S]) / A[4]
        
        s = acc
    
    return s
