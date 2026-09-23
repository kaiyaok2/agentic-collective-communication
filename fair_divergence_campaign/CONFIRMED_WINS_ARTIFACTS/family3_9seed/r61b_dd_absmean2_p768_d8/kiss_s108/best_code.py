def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    # Start with reshaped tensor
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 5 iterations with per-batch division
    for iteration in range(5):
        f_tensor = 1.0 + 2.0 * s.abs().mean(dim=1)
        f_unsqueezed = f_tensor.unsqueeze(1)
        
        # All_reduce requires flattened tensor
        buf = (s * f_unsqueezed).view(-1)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf).view(B, S)
        
        s = acc / (world_size * f_unsqueezed)
    
    # 6th iteration with simple division
    f_tensor = 1.0 + 2.0 * s.abs().mean(dim=1)
    buf = (s * f_tensor.unsqueeze(1)).view(-1)
    
    return xm.all_reduce(xm.REDUCE_SUM, buf) / world_size