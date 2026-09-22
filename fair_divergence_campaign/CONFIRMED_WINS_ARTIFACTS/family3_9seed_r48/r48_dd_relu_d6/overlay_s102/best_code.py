def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline sequential all-reduce chain with 6 collective dispatches.
    Each iteration: all_reduce, compute block means, apply relu scaling.
    """
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce: sum input across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iteration 1
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 2
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 3
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 4
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    
    # Iteration 5
    f = []
    for b in range(B):
        mb = s[b*S:(b+1)*S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    s = acc
    
    return s