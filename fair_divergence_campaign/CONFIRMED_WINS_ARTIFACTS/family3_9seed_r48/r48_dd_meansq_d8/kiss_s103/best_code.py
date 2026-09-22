def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for _ in range(6):
        f = (1.0 + (s * s).mean(dim=1)).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(B, S) / (world_size * f)
    
    f = (1.0 + (s * s).mean(dim=1)).unsqueeze(1)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size