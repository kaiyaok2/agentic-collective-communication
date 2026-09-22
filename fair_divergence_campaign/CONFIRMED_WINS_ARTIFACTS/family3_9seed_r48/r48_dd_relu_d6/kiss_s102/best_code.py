def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 4 iterations with factor normalization
    for iteration in range(4):
        # Vectorized mean computation
        s_reshaped = s.view(B, S)
        means = s_reshaped.mean(dim=1)
        
        # Compute factors
        f = []
        for b in range(B):
            mb = means[b]
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        buf = torch.zeros_like(s)
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for b in range(B):
            s[b*S:(b+1)*S] = s[b*S:(b+1)*S] / (world_size * f[b])
    
    # 5th iteration
    s_reshaped = s.view(B, S)
    means = s_reshaped.mean(dim=1)
    f = []
    for b in range(B):
        mb = means[b]
        f.append(1.0 + (mb if mb > 0 else mb*0.0))
    
    buf = torch.zeros_like(s)
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
    
    return s