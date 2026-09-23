def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)
    
    # First 6 iterations with factor division
    for iteration in range(6):
        f = 1.0 + s.abs().mean(dim=1, keepdim=True)
        buf = s * f
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        s = acc.view(B, S) / (world_size * f)
    
    # 7th iteration (different ending - no factor division)
    f = 1.0 + s.abs().mean(dim=1, keepdim=True)
    buf = s * f
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    s = acc / world_size
    
    return s