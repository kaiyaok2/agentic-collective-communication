
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
    ko = [(so + j) % 5 for j in range(2)]
    wd = 0.8 + 0.02*rank
    wo = 0.18 + 0.01*(rank % 5)
    
    for _ in range(6):
        buf = wd * s
        if len(ko) >= 1 and ko[0] >= 1:
            buf[ko[0]*S:(ko[0]+1)*S] = buf[ko[0]*S:(ko[0]+1)*S] + wo * s[(ko[0]-1)*S:ko[0]*S]
        if len(ko) >= 2 and ko[1] >= 1:
            buf[ko[1]*S:(ko[1]+1)*S] = buf[ko[1]*S:(ko[1]+1)*S] + wo * s[(ko[1]-1)*S:ko[1]*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Solve tridiagonal using separate blocks then concatenate
        rec0 = acc[0:S] / A[0]
        rec1 = (acc[S:2*S] - C[1] * rec0) / A[1]
        rec2 = (acc[2*S:3*S] - C[2] * rec1) / A[2]
        rec3 = (acc[3*S:4*S] - C[3] * rec2) / A[3]
        rec4 = (acc[4*S:5*S] - C[4] * rec3) / A[4]
        s = torch.cat([rec0, rec1, rec2, rec3, rec4], dim=0)
    
    # Final iteration
    buf = wd * s
    if len(ko) >= 1 and ko[0] >= 1:
        buf[ko[0]*S:(ko[0]+1)*S] = buf[ko[0]*S:(ko[0]+1)*S] + wo * s[(ko[0]-1)*S:ko[0]*S]
    if len(ko) >= 2 and ko[1] >= 1:
        buf[ko[1]*S:(ko[1]+1)*S] = buf[ko[1]*S:(ko[1]+1)*S] + wo * s[(ko[1]-1)*S:ko[1]*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
