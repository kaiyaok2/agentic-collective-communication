def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    ws = world_size
    
    for iteration in range(6):
        means = s.mean(dim=1, keepdim=True)
        f = 1.0 + means * means
        buf = (s * f).view(-1)
        s = xm.all_reduce(xm.REDUCE_SUM, buf).view(B, S) / (ws * f)
    
    means = s.mean(dim=1, keepdim=True)
    f = 1.0 + means * means
    buf = (s * f).view(-1)
    
    return xm.all_reduce(xm.REDUCE_SUM, buf) / ws