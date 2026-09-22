def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    ws = float(world_size)
    
    for i in range(7):
        f = (1.0 + (s * s).mean(dim=1)).unsqueeze(1)
        buf = s * f
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        if i < 6:
            s = s.view(B, S) / (ws * f)
        else:
            s = s / ws
    
    return s
