def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations
    for _ in range(6):
        f = 1.0 + (s * s).mean(dim=1, keepdim=True)
        s = xm.all_reduce(xm.REDUCE_SUM, s * f) / (world_size * f)
    
    # 7th iteration
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    
    return xm.all_reduce(xm.REDUCE_SUM, s * f).view(-1) / world_size
