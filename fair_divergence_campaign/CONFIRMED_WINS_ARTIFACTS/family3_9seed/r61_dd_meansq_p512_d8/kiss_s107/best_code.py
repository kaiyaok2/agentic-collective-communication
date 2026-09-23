def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s2d = s.view(B, S)
    
    # Perform 6 iterations
    for _ in range(6):
        f = 1.0 + (s2d * s2d).mean(dim=1, keepdim=True)
        s2d = xm.all_reduce(xm.REDUCE_SUM, (s2d * f).view(-1)).view(B, S) / (world_size * f)
    
    # Final iteration
    f = 1.0 + (s2d * s2d).mean(dim=1, keepdim=True)
    s = xm.all_reduce(xm.REDUCE_SUM, (s2d * f).view(-1)) / world_size
    
    return s