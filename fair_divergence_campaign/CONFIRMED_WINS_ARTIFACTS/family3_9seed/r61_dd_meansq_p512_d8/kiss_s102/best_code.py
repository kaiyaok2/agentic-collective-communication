def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Initial all_reduce to get synchronized s
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # The 6 intermediate all_reduce(s)/world_size operations are identity when s is synchronized
    # So we can skip directly to iteration 6
    
    # Iteration 6: scale by f
    f = 1.0 + (s * s).mean(dim=1, keepdim=True)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size