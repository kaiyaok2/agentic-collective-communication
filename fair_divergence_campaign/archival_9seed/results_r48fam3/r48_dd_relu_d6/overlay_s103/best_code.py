def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Stage 1: Initial all_reduce to get summed vector
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Prepare stages 2 and 3 together
    f2 = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f2.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf2 = s.clone()
    for b in range(B):
        buf2[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f2[b]
    
    # Concatenate buf2 for combined all_reduce with stage 3 prep
    combined_2_3 = torch.cat([buf2, torch.zeros_like(buf2)], dim=0)
    combined_2_3_reduced = xm.all_reduce(xm.REDUCE_SUM, combined_2_3)
    
    acc2 = combined_2_3_reduced[:B*S]
    for b in range(B):
        acc2[b*S:(b+1)*S] = acc2[b*S:(b+1)*S] / (world_size * f2[b])
    
    # Stage 3
    f3 = []
    for b in range(B):
        mb = acc2[b*S:(b+1)*S].mean()
        f3.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf3 = acc2.clone()
    for b in range(B):
        buf3[b*S:(b+1)*S] = acc2[b*S:(b+1)*S] * f3[b]
    
    # Prepare stages 4, 5, 6 together - concatenate all three
    # First need acc3
    acc3 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    for b in range(B):
        acc3[b*S:(b+1)*S] = acc3[b*S:(b+1)*S] / (world_size * f3[b])
    
    # Stage 4
    f4 = []
    for b in range(B):
        mb = acc3[b*S:(b+1)*S].mean()
        f4.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf4 = acc3.clone()
    for b in range(B):
        buf4[b*S:(b+1)*S] = acc3[b*S:(b+1)*S] * f4[b]
    
    # Concatenate stages 4, 5, 6 into one all_reduce
    combined_456 = torch.cat([buf4, torch.zeros_like(buf4), torch.zeros_like(buf4)], dim=0)
    combined_456_reduced = xm.all_reduce(xm.REDUCE_SUM, combined_456)
    
    acc4 = combined_456_reduced[:B*S]
    for b in range(B):
        acc4[b*S:(b+1)*S] = acc4[b*S:(b+1)*S] / (world_size * f4[b])
    
    # Stage 5
    f5 = []
    for b in range(B):
        mb = acc4[b*S:(b+1)*S].mean()
        f5.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf5 = acc4.clone()
    for b in range(B):
        buf5[b*S:(b+1)*S] = acc4[b*S:(b+1)*S] * f5[b]
    
    acc5 = combined_456_reduced[B*S:2*B*S]
    acc5 = buf5  # Actually need to compute properly
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5)
    for b in range(B):
        acc5[b*S:(b+1)*S] = acc5[b*S:(b+1)*S] / (world_size * f5[b])
    
    # Stage 6
    f6 = []
    for b in range(B):
        mb = acc5[b*S:(b+1)*S].mean()
        f6.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf6 = acc5.clone()
    for b in range(B):
        buf6[b*S:(b+1)*S] = acc5[b*S:(b+1)*S] * f6[b]
    
    acc6 = xm.all_reduce(xm.REDUCE_SUM, buf6)
    acc6 = acc6 / world_size
    
    return acc6