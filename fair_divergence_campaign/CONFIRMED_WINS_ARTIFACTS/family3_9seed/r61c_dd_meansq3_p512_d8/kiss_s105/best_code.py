
def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    # First 6 iterations with per-batch division
    for _ in range(6):
        mean_sq = (s * s).mean(dim=1)
        f = (1.0 + 3.0 * mean_sq).unsqueeze(1)
        
        buf = (s * f).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc.view(B, S) / (world_size * f)
    
    # 7th iteration with simple division
    mean_sq = (s * s).mean(dim=1)
    f = (1.0 + 3.0 * mean_sq).unsqueeze(1)
    buf = (s * f).view(-1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = (acc / world_size).view(-1)
    
    return s
