def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x.reshape(B, S))
    
    for iteration in range(6):
        f = 1.0 + (s * s).mean(dim=1, keepdim=True)
        acc = xm.all_reduce(xm.REDUCE_SUM, s * f)
        s = acc / (world_size * f)
    
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    acc = xm.all_reduce(xm.REDUCE_SUM, s * f)
    s = acc / world_size
    
    return s.flatten()