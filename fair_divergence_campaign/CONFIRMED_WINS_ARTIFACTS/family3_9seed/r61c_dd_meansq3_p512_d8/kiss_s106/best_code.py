def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations
    for iteration in range(6):
        # Compute factors: f[b] = 1.0 + 3.0 * mean(s[b]**2)
        f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)  # [B, 1]
        
        # Multiply, all_reduce, divide
        buf = s * f
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        s = acc / (world_size * f)
    
    # Last iteration
    f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)
    buf = s * f
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size