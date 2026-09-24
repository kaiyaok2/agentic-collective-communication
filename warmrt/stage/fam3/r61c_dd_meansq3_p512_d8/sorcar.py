def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(8, 512)
    
    # First 6 iterations
    for _ in range(6):
        f = (1.0 + 3.0 * (s * s).mean(dim=1)).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(8, 512) / (world_size * f)
    
    # Last iteration
    f = (1.0 + 3.0 * (s * s).mean(dim=1)).unsqueeze(1)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size