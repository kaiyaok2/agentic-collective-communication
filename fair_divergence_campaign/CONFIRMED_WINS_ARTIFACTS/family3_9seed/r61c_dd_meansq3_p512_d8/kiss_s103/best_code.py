def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Initial all-reduce and reshape once
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations with full normalization
    for iter in range(6):
        mean_sq = (s * s).mean(dim=1)
        f = (1.0 + 3.0 * mean_sq).unsqueeze(1)
        buf = s * f
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        s = acc / (world_size * f)
    
    # 7th iteration without factor normalization
    mean_sq = (s * s).mean(dim=1)
    f = 1.0 + 3.0 * mean_sq
    buf = s * f.unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size