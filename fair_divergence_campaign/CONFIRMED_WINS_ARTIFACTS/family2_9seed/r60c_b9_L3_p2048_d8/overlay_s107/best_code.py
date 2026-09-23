def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # Precompute scaling factors for each block
    c = [0] * 9
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(3))
        for b in ks:
            c[b] += 1
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(3))
    
    # Pipeline the 7 selective all_reduce stages
    # Stage 1: Process blocks in keep set, prepare buffer
    buf1 = torch.zeros_like(s)
    for b in keep:
        buf1[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    for b in range(9):
        acc1[b*S:(b+1)*S] = acc1[b*S:(b+1)*S] / c[b]
    
    # Stage 2
    buf2 = torch.zeros_like(acc1)
    for b in keep:
        buf2[b*S:(b+1)*S] = acc1[b*S:(b+1)*S]
    acc2 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    for b in range(9):
        acc2[b*S:(b+1)*S] = acc2[b*S:(b+1)*S] / c[b]
    
    # Stage 3
    buf3 = torch.zeros_like(acc2)
    for b in keep:
        buf3[b*S:(b+1)*S] = acc2[b*S:(b+1)*S]
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    for b in range(9):
        acc3[b*S:(b+1)*S] = acc3[b*S:(b+1)*S] / c[b]
    
    # Stage 4
    buf4 = torch.zeros_like(acc3)
    for b in keep:
        buf4[b*S:(b+1)*S] = acc3[b*S:(b+1)*S]
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4)
    for b in range(9):
        acc4[b*S:(b+1)*S] = acc4[b*S:(b+1)*S] / c[b]
    
    # Stage 5
    buf5 = torch.zeros_like(acc4)
    for b in keep:
        buf5[b*S:(b+1)*S] = acc4[b*S:(b+1)*S]
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    for b in range(9):
        acc5[b*S:(b+1)*S] = acc5[b*S:(b+1)*S] / c[b]
    
    # Stage 6
    buf6 = torch.zeros_like(acc5)
    for b in keep:
        buf6[b*S:(b+1)*S] = acc5[b*S:(b+1)*S]
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    for b in range(9):
        acc6[b*S:(b+1)*S] = acc6[b*S:(b+1)*S] / c[b]
    
    # Stage 7 (final, no scaling needed after)
    buf7 = torch.zeros_like(acc6)
    for b in keep:
        buf7[b*S:(b+1)*S] = acc6[b*S:(b+1)*S]
    acc7 = xm.all_reduce(xm.REDUCE_SUM, buf7)
    
    return acc7