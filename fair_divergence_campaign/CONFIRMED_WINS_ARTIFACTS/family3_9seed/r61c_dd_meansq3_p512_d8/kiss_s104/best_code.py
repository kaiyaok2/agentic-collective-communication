def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Initial all_reduce and reshape to batched form
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Do first 6 iterations
    for _ in range(6):
        f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(B, S)
        s = acc / (world_size * f)
    
    # Do last iteration (different division)
    f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)
    acc = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1))
    
    return acc / world_size